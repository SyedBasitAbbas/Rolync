import os
import asyncio
import logging
import uvicorn
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
import json
import re

# Load environment variables from .env file in the V38 directory
load_dotenv(Path(__file__).parent / ".env")

# --- Local Module Imports ---
from models import (
    ChatRequest, ChatResponse, SessionState, MatchingRequest, MatchingResponse,
    SummaryRetrievalResponse, InterviewState, DetailedEvaluation
)
from session_manager import get_session, save_session
from database import get_user_collection, get_referrers_collection, close_db_connection
from agents import profiling_agent, interview_agent, doubt_agent

# --- Constants & Configuration ---
COLORS = {
    'YELLOW': '\033[93m', 'GREEN': '\033[92m', 'RED': '\033[91m',
    'CYAN': '\033[96m', 'RESET': '\033[0m'
}

# --- Logging Setup ---
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        message = super().format(record)
        return f"{COLORS['CYAN']}{message}{COLORS['RESET']}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logging.getLogger("pymongo").setLevel(logging.WARNING)
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)

# --- Gemini API Setup ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is required.")

# Initialize the client globally (rather than configuring genai)
gemini_client = genai.Client(api_key=gemini_api_key)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the ROLync API. Please use the /docs endpoint for documentation."}

@app.post("/user/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """ Main chat endpoint. Manages the conversation flow based on the session's current stage. """
    user_collection = get_user_collection()
    session = get_session(request.session_id, user_collection)
    logger.info(f"Chat request for session '{session.session_id}' in stage '{session.current_stage}'")

    bot_reply = ""
    
    if session.current_stage == 'profiling':
        bot_reply = await profiling_agent.process(request.user_input, session)
    elif session.current_stage == 'interviewing':
        if not session.interview_state.conversation_history:
             logger.info(f"First message in interview stage for session {session.session_id}: '{request.user_input}'")
        
        # Store the previous session stage to detect transitions
        previous_stage = session.current_stage
        
        # Process the user's message
        bot_reply = await interview_agent.process(request.user_input, session)
        
        # Check if this was the final answer that triggered completion
        if previous_stage == 'interviewing' and session.current_stage == 'doubts':
            logger.info(f"Detected transition from interviewing to doubts stage for session {session.session_id}")
            
            # Check if matches are already present
            if session.matches:
                logger.info(f"Session already has {len(session.matches)} matches - no need to generate")
            else:
                # Generate matches immediately
                logger.info(f"Interview completed, generating matches immediately for session {session.session_id}")
                matches = await interview_agent._find_matches(session)
                
                if matches:
                    # Set matches in session state
                    session.matches = matches
                    logger.info(f"Generated {len(matches)} matches for session {session.session_id}")
                    
                    # Force save directly to MongoDB
                    update_result = user_collection.update_one(
                        {"session_id": session.session_id},
                        {"$set": {
                            "matches": matches,
                            "current_stage": "doubts"
                        }}
                    )
                    logger.info(f"Direct MongoDB update result: modified_count={update_result.modified_count}")
                    
                    # Verify the update
                    verification = user_collection.find_one({"session_id": session.session_id})
                    if verification and "matches" in verification:
                        matches_in_db = verification["matches"]
                        logger.info(f"Verification: Found {len(matches_in_db)} matches in UserDataCollection")
                    else:
                        logger.error(f"Verification failed: Matches not found in UserDataCollection after update")
    elif session.current_stage == 'doubts':
        # Initialize doubt_session_start if not set
        if not session.doubt_session_start:
            session.doubt_session_start = datetime.now(timezone.utc)
        
        # Check if rate limit has been reached
        if session.doubt_queries_count >= session.doubt_queries_limit:
            bot_reply = "You've reached the maximum number of follow-up questions for this session."
        else:
            # Store user's question in conversation history
            session.doubt_conversation_history.append({"role": "user", "content": request.user_input})
            
            # Process the question with DoubtAgent
            session.doubt_queries_count += 1
            bot_reply = await doubt_agent.process(request.user_input, session)
            
            # Store agent's response in conversation history
            session.doubt_conversation_history.append({"role": "assistant", "content": bot_reply})
            
            # Log the query count
            logger.info(f"Session {session.session_id} doubt query {session.doubt_queries_count}/{session.doubt_queries_limit}")
            logger.info(f"Doubt conversation history now contains {len(session.doubt_conversation_history)} messages")
    else: # 'summary' or 'complete'
        bot_reply = "Our interview session is complete. You can ask questions about your evaluation."
        
        # If stage is 'complete' and should be 'doubts', fix it
        if session.current_stage == 'complete' and session.summary_data:
            logger.info(f"Correcting session stage from 'complete' to 'doubts' for session {session.session_id}")
            session.current_stage = 'doubts'
            
            # Direct update to ensure stage is fixed
            user_collection.update_one(
                {"session_id": session.session_id},
                {"$set": {"current_stage": "doubts"}}
            )

    # Save session and ensure 'matches' field is properly set in MongoDB
    if session.matches:
        logger.info(f"Ensuring {len(session.matches)} matches are saved in UserDataCollection")
        
    save_session(session, user_collection)
    return ChatResponse(session_id=session.session_id, message=bot_reply)

@app.get("/user/evaluation/{session_id}/{question_id}", response_model=DetailedEvaluation)
async def get_evaluation_details(session_id: str, question_id: str):
    """Retrieves the detailed evaluation for a specific question."""
    session = get_session(session_id, get_user_collection())
    
    # Search for the evaluation with matching question_id
    matching_eval = None
    for eval in session.detailed_evaluations:
        if eval.question_id == question_id:
            matching_eval = eval
            break
    
    if matching_eval:
        return matching_eval
    else:
        raise HTTPException(status_code=404, detail=f"No evaluation found for question ID {question_id} in session {session_id}")

@app.get("/user/summary/{session_id}", response_model=SummaryRetrievalResponse)
async def get_interview_summary(session_id: str):
    """Retrieves the interview summary for a session."""
    session = get_session(session_id, get_user_collection())
    
    if (session.current_stage == 'complete' or session.current_stage == 'doubts') and session.summary_data:
        return SummaryRetrievalResponse(session_id=session_id, summary=session.summary_data, status="ready", message="Summary is ready.")
    elif session.current_stage in ['interviewing', 'summary']:
        return SummaryRetrievalResponse(session_id=session_id, summary=None, status="processing", message="Summary is still being generated.")
    else:
        return SummaryRetrievalResponse(session_id=session_id, summary=None, status="not_started", message="Interview not yet complete or not started.")

@app.post("/agent/match", response_model=MatchingResponse)
async def handle_matching(request: MatchingRequest):
    """Process matching request and save matched profiles in UserDataCollection."""
    logger.info(f"Matching endpoint called for session {request.session_id}")
    
    try:
        # Get the session from UserDataCollection
        user_collection = get_user_collection()
        session = get_session(request.session_id, user_collection)
        
        # If the session already has matches, return them
        if session.matches and len(session.matches) > 0:
            logger.info(f"Found {len(session.matches)} existing matches for session {request.session_id}")
            return MatchingResponse(session_id=request.session_id, matches=session.matches)
        
        # If session is in doubts or complete stage but has no matches, try to generate them
        if session.current_stage == 'complete' or session.current_stage == 'doubts':
            # If we have summary data, we can generate matches
            if session.summary_data:
                matches = await interview_agent._find_matches(session)
                if matches:
                    # Save matches to the session state
                    session.matches = matches
                    save_session(session, user_collection)
                    logger.info(f"Generated and saved {len(matches)} matches to UserDataCollection for session {request.session_id}")
                    return MatchingResponse(session_id=request.session_id, matches=matches)
        
        # If session is not complete/doubts or no matches were found
        return MatchingResponse(
            session_id=request.session_id, 
            matches=[{"detail": "No matches available yet. Please complete the interview first."}]
        )
        
    except Exception as e:
        logger.error(f"Error in matching endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing matching request: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    close_db_connection()

# For local development only - not used in production/Vercel
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False) 


