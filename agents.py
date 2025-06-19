# Placeholder for agents 

import os
import asyncio
import re
import time
import functools
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

# --- Local Imports ---
from models import SessionState, InterviewState
from database import get_user_collection, close_db_connection
from session_manager import save_session

logger = logging.getLogger(__name__)

# --- Utility Functions ---

def split_into_paragraphs(text: str) -> List[str]:
    """ Splits text into meaningful paragraphs for processing. """
    if not text: return []
    paragraphs = re.split(r'\n\s*\n', text)
    substantial_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 150 and any(char.isalpha() for char in p)]
    return substantial_paragraphs if substantial_paragraphs else [text]

# --- Core Agent Logic ---

class ProfilingAgent:
    """
    Handles the initial user profile data collection using a conversational,
    LLM-driven approach with validation.
    """
    def __init__(self):
        self.fields = ["name", "target_company", "job_title", "user_experience", "user_long_term_objective"]
        self.field_prompts = {
            "name": "Ask for the user's full name in a friendly, professional tone.",
            "target_company": "Ask for the name of the company the user is interested in.",
            "job_title": "Ask for the specific job title the user is seeking (e.g., 'AI Engineer', 'Product Manager').",
            "user_experience": "Ask for a brief summary of the user's relevant work experience or background.",
            "user_long_term_objective": "Ask about the user's long-term career objectives or aspirations."
        }
        # Check if environment variable is set
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Initialize client instead of GenerativeModel
        self.client = genai.Client(api_key=api_key)
        # We'll use this model ID directly in methods, no need to store model object
        self.model_name = "gemini-2.5-flash-preview-04-17"

    async def _call_llm(self, prompt: str, is_json: bool = False) -> str:
        """Helper method to call the LLM with proper settings."""
        try:
            # Create appropriate content structure
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation parameters
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json" if is_json else "text/plain",
                temperature=0,
            )
            
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            return response.text.strip()
        except Exception as e:
            logging.error(f"LLM call failed: {str(e)}", exc_info=True)
            if "rate limit" in str(e).lower():
                return '{"error": "API rate limit exceeded. Please try again in a moment."}' if is_json else "I'm currently experiencing high demand. Could you try again in a moment?"
            return '{"error": "Failed to get a valid response from the model."}' if is_json else "I'm having a little trouble processing that. Could we try again?"

    async def _validate_and_extract_response(self, field: str, response: str, state: SessionState) -> dict:
        """
        Validates user response, extracts data, and detects intent (e.g., clarification questions).
        Returns a dictionary with intent, validity, extracted value, and a generated AI response.
        """
        conversation_history = state.profiling_state.conversation_history
        formatted_history = ""
        if len(conversation_history) > 1:
            formatted_history = "Previous conversation:\n"
            for entry in conversation_history[-10:]:
                if "user" in entry: formatted_history += f"User: {entry['user']}\n"
                if "assistant" in entry: formatted_history += f"Assistant: {entry['assistant']}\n"
        
        prompt = f"""
        You are an intelligent conversational AI assistant helping a user build their professional profile.

        **CONTEXT:**
        - You just asked the user for: "{state.profiling_state.last_question}" (which corresponds to the '{field.replace('_', ' ')}' field).
        - The user's latest response is: "{response}"
        - Here is the recent conversation history:
        {formatted_history}
        - Here is the data collected so far: {json.dumps(state.profiling_state.user_data)}

        **YOUR TASKS:**

        1.  **Analyze Intent**: First, determine the user's primary intent. Choose one:
            *   `answering`: The user is providing information directly related to the '{field}' you asked for.
            *   `clarification`: The user is asking a question to better understand what information you need (e.g., "what do you mean?", "what kind of background?").
            *   `unrelated`: The user's response is off-topic or a general question not about the current field.

        2.  **Act Based on Intent**:
            *   **If intent is `answering`**:
                - **Synthesize & Extract**: THIS IS A CRITICAL STEP. Review the user's latest response in the context of the entire conversation history. You MUST combine information to form the most complete value. For example, if the history mentions the user is interested in a "Data Scientist" role and their latest response is "Senior", you must synthesize this to extract "Senior Data Scientist" for the `job_title` field.
                - **Validate**: Determine if the synthesized value is a plausible and sufficient answer for the '{field}' field.
                - **Create Response**: If the answer is invalid (e.g., too short, irrelevant), create a helpful `ai_response` explaining what you need. If it's valid, the `ai_response` can be a simple, brief confirmation like "Got it."

            *   **If intent is `clarification`**:
                - Create a helpful `ai_response` that answers their clarification question and re-asks for the information in a clearer way. Use the field description for context: "{self.field_prompts[field]}"
            *   **If intent is `unrelated`**:
                - Create a polite `ai_response` that gently redirects them back to the current question.

        **OUTPUT FORMAT:**
        Return a single JSON object with the following exact structure. DO NOT add extra text or formatting.

        {{
          "intent": "answering | clarification | unrelated",
          "is_valid": true | false,
          "extracted_value": "The complete, synthesized data, or null if not applicable",
          "ai_response": "The helpful, user-facing response you crafted."
        }}

        **Example for this request:**
        If the user says "what kind of background you need?" for the 'user_experience' field, your output should be:
        {{
            "intent": "clarification",
            "is_valid": false,
            "extracted_value": null,
            "ai_response": "I'm looking for a brief summary of your work history, key projects, or skills that are relevant to the {state.profiling_state.user_data.get('job_title', 'role')} you're interested in. For example, you could mention the last couple of roles you've held or a significant achievement."
        }}
        """
        llm_response_str = await self._call_llm(prompt, is_json=True)
        try:
            cleaned_text = re.sub(r"```json\s*|\s*```", "", llm_response_str)
            result = json.loads(cleaned_text)
            logger.info(f"Validation/Intent result for '{field}': {result}")
            return result
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to decode validation JSON: {llm_response_str}. Error: {e}")
            return {
                "intent": "answering", "is_valid": False, "extracted_value": None,
                "ai_response": "I had a little trouble understanding that. Could you please rephrase your response?"
            }

    async def _generate_question(self, field: str, state: SessionState) -> str:
        """Generates the next question to ask the user."""
        user_name_pleasantry = f"Nice to meet you, {state.profiling_state.user_data.get('name', 'there')}! " if field == 'target_company' and 'name' in state.profiling_state.user_data else ""
        
        # Get conversation history
        conversation_history = state.profiling_state.conversation_history
        
        # Create a formatted conversation history string
        formatted_history = ""
        if conversation_history:
            formatted_history = "Previous conversation:\n"
            # Limit to the last 5 exchanges to keep context manageable
            for entry in conversation_history[-10:]:  # Use last 10 exchanges
                if "user" in entry:
                    formatted_history += f"User: {entry['user']}\n"
                if "assistant" in entry:
                    formatted_history += f"Assistant: {entry['assistant']}\n"

        # ProfilingAgent prompt: Generates conversational questions to collect user profile data
        prompt = f"""
        You are a friendly career assistant.
        {user_name_pleasantry}Generate the next question to ask the user for their '{field.replace('_', ' ')}'.
        This is the official description of what to ask: "{self.field_prompts[field]}"
        
        {formatted_history}
        
        Current collected data: {json.dumps(state.profiling_state.user_data)}
        
        IMPORTANT GUIDELINES:
        - Keep it concise and conversational
        - Do not repeat greetings
        - If the user has already mentioned information about '{field}' in previous exchanges, acknowledge that and ask for clarification or additional details
        - Make your question natural and flowing from the conversation context
        - Avoid asking for information that the user has already provided
        """
        return await self._call_llm(prompt)

    async def process(self, user_input: str, state: SessionState) -> str:
        """Main processing logic for the profiling conversation."""
        profiling = state.profiling_state
        
        profiling.conversation_history.append({"user": user_input})

        if not profiling.last_question:
            current_field = self.fields[0]
            reply = await self._generate_question(current_field, state)
        else:
            current_field_index = profiling.current_field_index
            
            if current_field_index >= len(self.fields):
                return "Our profiling session is complete. We are now moving to the interview phase."

            current_field = self.fields[current_field_index]
            
            validation_result = await self._validate_and_extract_response(current_field, user_input, state)
            
            intent = validation_result.get("intent")
            is_valid = validation_result.get("is_valid", False)
            
            if intent == 'answering' and is_valid:
                # Valid answer received, store data and move to the next field
                profiling.user_data[current_field] = validation_result.get("extracted_value")
                next_field_index = current_field_index + 1
                profiling.current_field_index = next_field_index

                if next_field_index >= len(self.fields):
                    state.current_stage = 'interviewing'
                    asyncio.create_task(interview_agent.prepare_interview(state))
                    reply = "Great, thanks! I have everything for your profile. We'll now move on to a short practice interview. I'll ask a few questions based on the role and company you mentioned. Are you ready?"
                else:
                    next_field = self.fields[next_field_index]
                    reply = await self._generate_question(next_field, state)
            else:
                # Handle clarification, invalid answer, or unrelated query
                # The LLM has already generated the appropriate response.
                # We do not advance to the next field.
                reply = validation_result.get("ai_response", "I'm not sure how to handle that. Could you try rephrasing?")

        profiling.last_question = reply
        profiling.conversation_history.append({"assistant": reply})
        
        logger.info(f"ProfileAgent - Current field: {self.fields[profiling.current_field_index] if profiling.current_field_index < len(self.fields) else 'None'}, Field index: {profiling.current_field_index}, User data: {json.dumps(profiling.user_data)}")
        
        return reply


class InterviewAgent:
    """ Handles the Q&A interview flow, including search, question generation, evaluation, and summary. """
    MAX_QUESTIONS = 3

    def __init__(self):
        # Check if environment variable is set
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Initialize client instead of GenerativeModel
        self.client = genai.Client(api_key=api_key)
        # We'll use this model ID directly in methods, no need to store model object
        self.model_name = "gemini-2.5-flash-preview-04-17"

    async def _call_llm(self, prompt: str, is_json: bool = False) -> str:
        """Helper method to call the LLM with proper settings."""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json" if is_json else "text/plain",
                temperature=0,
            )
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            return response.text.strip()
        except Exception as e:
            logging.error(f"LLM call failed: {str(e)}", exc_info=True)
            if "rate limit" in str(e).lower():
                return '{"error": "API rate limit exceeded. Please try again in a moment."}' if is_json else "I'm currently experiencing high demand. Could you try again in a moment?"
            return '{"error": "Failed to get a valid response from the model."}' if is_json else "I'm having a little trouble processing that. Could we try again?"

    async def prepare_interview(self, state: SessionState):
        """BACKGROUND TASK: Runs search agent to gather data for the interview."""
        if state.search_state.search_data:
            logger.info(f"Search data already exists for session {state.session_id}. Skipping search.")
            return
        
        # Ensure interview state has paragraph_question_counter initialized
        state.interview_state.paragraph_question_counter = 0
        state.interview_state.current_paragraph_index = 0
        
        logger.info(f"Starting background search for session {state.session_id}...")
        await self._search(state)
        logger.info(f"Background search completed for session {state.session_id}")
        
        # Save the updated session with search data
        save_session(state, get_user_collection())

    async def _detect_intent(self, user_input: str, question: str) -> dict:
        """Detects whether the user is answering the question or asking a question/clarification."""
        
        prompt = f"""
        Analyze the user's message in relation to the interview question they were asked.
        
        INTERVIEW QUESTION: "{question}"
        
        USER'S RESPONSE: "{user_input}"
        
        Determine if the user is:
        1. Providing an answer to the question
        2. Asking for clarification about the question
        3. Asking a completely different question
        4. Providing incomplete or unclear information
        
        Return a JSON object with this structure:
        {{
            "intent": "answer|clarification|question|unclear",
            "confidence": 0.0 to 1.0,
            "explanation": "Brief explanation of the classification"
        }}
        """
        
        try:
            # Create appropriate content structure
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation parameters
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            )
            
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            response_text = response.text.strip()
            # Clean up JSON string if needed
            cleaned_json = re.sub(r"```json\s*|\s*```", "", response_text)
            result = json.loads(cleaned_json)
            
            logger.info(f"Intent detection result: {result['intent']} (confidence: {result['confidence']})")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting intent: {e}", exc_info=True)
            # Default to assuming it's an answer if detection fails
            return {
                "intent": "answer",
                "confidence": 0.5,
                "explanation": "Failed to properly detect intent, defaulting to answer"
            }
    
    async def _handle_clarification(self, user_input: str, question: str) -> str:
        """Generate a response to a user's request for clarification about the question."""
        
        prompt = f"""
        The user was asked the following interview question:
        "{question}"
        
        But instead of answering directly, they asked for clarification:
        "{user_input}"
        
        Provide a helpful clarification that:
        1. Addresses their specific confusion or request
        2. Provides additional context about what the question is asking
        3. Encourages them to answer the original question
        4. Maintains a professional, supportive tone
        
        Keep your response concise and focused on helping them understand what information you're seeking.
        """
        
        try:
            # Create appropriate content structure
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation parameters
            gen_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.3,
            )
            
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating clarification: {e}", exc_info=True)
            return "I'm looking for information about your relevant work experience for this role. Could you share some details about your background that relate to this position?"

    async def process(self, user_input: str, state: SessionState) -> str:
        """ Processes a user's answer and returns the next question or completes the interview. """
        interview = state.interview_state
        detailed_eval = None
        
        # Ensure search data is available - if not, wait for it
        if not state.search_state.search_data:
            # Wait for search data to be available (max 60 seconds)
            logger.info(f"Search data not ready for session {state.session_id}, waiting for completion")
            max_wait_time = 60  # seconds
            wait_interval = 0.5  # seconds
            for _ in range(int(max_wait_time / wait_interval)):
                await asyncio.sleep(wait_interval)
                # Re-check if search data is now available
                if state.search_state.search_data:
                    logger.info(f"Search data is now available after waiting")
                    break
            
            # If still no search data after waiting, generate a fallback question
            if not state.search_state.search_data:
                logger.warning(f"Search data still not available after waiting {max_wait_time} seconds")
                # Don't show a "still preparing" message - directly ask a general question
                next_question = "What are your key strengths as they relate to this role?"
                interview.questions.append(next_question)
                interview.conversation_history.append({"assistant": next_question})
                return next_question
            
            # If we have search data but haven't asked the first question yet, generate it now
            if not interview.questions:
                next_question = await self._ask_next_question(state)
                interview.conversation_history.append({"assistant": next_question})
                return next_question

        # First-time entry with ready search data
        if not interview.questions:
            next_question = await self._ask_next_question(state)
            interview.conversation_history.append({"assistant": next_question})
            return next_question

        # Track conversation history for all user messages
        interview.conversation_history.append({"user": user_input})
            
        # If we have a previous question, detect the user's intent
        if interview.questions:
            last_question = interview.questions[-1]
            intent_result = await self._detect_intent(user_input, last_question)
            
            # Handle different intents
            if intent_result["intent"] != "answer" and intent_result["confidence"] > 0.6:
                if intent_result["intent"] == "clarification":
                    # User is asking for clarification about the question
                    clarification = await self._handle_clarification(user_input, last_question)
                    interview.conversation_history.append({"assistant": clarification})
                    return clarification
                elif intent_result["intent"] == "question":
                    # User is asking an unrelated question - politely redirect
                    redirect = f"I understand you have a question, but I'm currently focused on conducting your interview. Let's continue with the current question: {last_question}"
                    interview.conversation_history.append({"assistant": redirect})
                    return redirect
                elif intent_result["intent"] == "unclear":
                    # User's response is unclear
                    prompt = f"I'm not sure I understand your response. Could you please provide more details in relation to the question: {last_question}"
                    interview.conversation_history.append({"assistant": prompt})
                    return prompt
            
            # If intent is "answer" or we're not confident in another classification, treat as answer
            interview.answers.append(user_input)
            
            # Evaluate the answer if this isn't the first message
            if len(interview.questions) >= 1:
                try:
                    evaluation_result = await self._evaluate_answer(state)
                    
                    # Store the evaluation score in the existing fields
                    if "score" in evaluation_result and evaluation_result["score"] is not None:
                        score = float(evaluation_result["score"])
                        # Consider scores of 0 as incorrect answers for difficulty adjustment
                        interview.scores.append(score)
                        logging.info(f"Recorded score: {score}")
                        
                    if "max_score" in evaluation_result and evaluation_result["max_score"] is not None:
                        max_score = float(evaluation_result["max_score"])
                        interview.max_scores.append(max_score)
                        logging.info(f"Recorded max score: {max_score}")
                    
                    # Get the latest detailed evaluation to include in response
                    if state.detailed_evaluations and len(state.detailed_evaluations) > 0:
                        detailed_eval = state.detailed_evaluations[-1]
                    
                except Exception as e:
                    logging.error(f"Error evaluating answer: {e}", exc_info=True)
                    # Add a default score to maintain consistency in the lists
                    interview.scores.append(1.0)  # Default to moderate score on error
                    interview.max_scores.append(2.0)
        
        # Check if we've reached the maximum number of questions
        if len(interview.questions) >= self.MAX_QUESTIONS:
            # Generate summary and transition to summary stage
            state.current_stage = 'summary'
            await self._generate_summary(state)
            state.current_stage = 'complete'
            final_message = "Thank you for completing the interview! I'll now analyze your responses and prepare a summary of your performance along with potential matches."
            interview.conversation_history.append({"assistant": final_message})
            
            # Find matches based on summary and return them to the user
            matches = await self._find_matches(state)
            
            # Format matches for display to user
            if matches:
                match_count = len(matches)
                match_details = "\n\nHere are potential referrers who match your profile:\n"
                for i, match in enumerate(matches[:5], 1):  # Show up to 5 matches
                    match_name = match.get("name", "Unnamed Referrer")
                    match_company = match.get("company", "Unknown Company")
                    match_role = match.get("role", "Unknown Role")
                    match_details += f"\n{i}. {match_name} - {match_company}, {match_role}"
                
                return f"Thanks for completing the interview! Based on your responses, I've found {match_count} potential referrals that match your profile.{match_details}\n\nYou can reach out to these professionals for referral opportunities."
            else:
                return "Thanks for completing the interview! I've analyzed your responses and am searching for potential referrals that match your profile."
        
        # Generate the next question
        next_question = await self._ask_next_question(state)
        
        # Note: We're removing the code that adds strengths/weaknesses to the bot reply
        # The evaluation still happens in the background and is stored in state.detailed_evaluations
        
        interview.conversation_history.append({"assistant": next_question})
        return next_question

    async def finalize_interview(self, state: SessionState):
        """BACKGROUND TASK: Runs final evaluation and generates summary."""
        logger.info(f"Finalizing interview for session {state.session_id}...")
        await self._generate_summary(state)
        
        # Find matches after generating summary
        logger.info(f"Finding matches for session {state.session_id}...")
        matches = await self._find_matches(state)
        
        # Log the raw matches to verify what's being returned
        if matches:
            logger.info(f"Generated {len(matches)} matches for session {state.session_id}, first match: {json.dumps(matches[0])}")
            
            # 1. Set matches in session state
            state.matches = matches
            logger.info("Matches set in session state")
            
            # 2. Force update UserDataCollection directly
            try:
                user_collection = get_user_collection()
                # First verify the session exists in the collection
                session_doc = user_collection.find_one({"session_id": state.session_id})
                if session_doc:
                    # Direct update to ensure matches are stored
                    update_result = user_collection.update_one(
                        {"session_id": state.session_id},
                        {"$set": {
                            "matches": matches,
                            "current_stage": "doubts"
                        }}
                    )
                    logger.info(f"Direct MongoDB update result: modified_count={update_result.modified_count}")
                    
                    # Verify the update was successful by retrieving the document
                    verification = user_collection.find_one({"session_id": state.session_id})
                    if verification and "matches" in verification:
                        matches_in_db = verification["matches"]
                        logger.info(f"Verification: Found {len(matches_in_db)} matches in the UserDataCollection document")
                    else:
                        logger.error(f"Verification failed: Matches not found in UserDataCollection after update")
                else:
                    logger.error(f"Session {state.session_id} not found in UserDataCollection")
            except Exception as e:
                logger.error(f"Error updating matches in UserDataCollection: {str(e)}", exc_info=True)
        else:
            logger.warning(f"No matches were generated for session {state.session_id}")
        
        # Set stage to doubts so the user can ask questions
        state.current_stage = 'doubts'
        
        # Save session one more time to ensure all changes are persisted
        user_collection = get_user_collection()
        save_session(state, user_collection)
        
        # Final verification that matches are stored
        try:
            verification = user_collection.find_one({"session_id": state.session_id})
            if verification and "matches" in verification and verification["matches"]:
                logger.info(f"Final verification: {len(verification['matches'])} matches confirmed in UserDataCollection")
            else:
                logger.error(f"Final verification failed: Matches not properly stored in UserDataCollection")
        except Exception as e:
            logger.error(f"Error in final verification: {str(e)}")
            
        logger.info(f"Summary and matches generated, session '{state.session_id}' marked as in 'doubts' stage.")

    async def _search(self, state: SessionState):
        """Perform search to gather information about the job role and company"""
        logging.info(f"Starting background search for session {state.session_id}...")
        
        # Extract profile data
        profile = state.profiling_state.user_data
        job_title = profile.get('job_title', 'the specified role')
        company = profile.get('target_company', 'the specified company')
        
        # InterviewAgent prompt: Uses Google Search to gather company and job role information
        search_prompt = f"""
        You are a Search Agent with access to the Google Search API. Your task is to gather detailed information about the company '{company}'
        and the job role '{job_title}'. 
        
        IMPORTANT: YOU MUST USE THE GOOGLE SEARCH TOOL to find recent and accurate information. DO NOT make up information.
        
        For each topic below, run at least 1-2 specific Google searches and include the information you found:
        
        1. Company background, mission, values, and recent news for {company}
        2. Products, services, and technologies used at {company}
        3. Common interview questions for {job_title} positions at {company}
        4. Skills and qualifications typically required for {job_title} at {company}
        5. Company culture and work environment at {company}
        6. Recent projects or innovations at {company}
        
        For each search query, paste the actual search query you use and then summarize the information found.
        
        Structure your response clearly with headings for each section. Include source URLs where relevant.
        
        REMEMBER: Use the Google Search tool FIRST before providing any information. Your response should ONLY include information you found through search results.
        """
        
        try:
            # Configure client and Google Search tool
            client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
            )
            
            # Create the tools using the correct format for Google Search
            tools = [
                types.Tool(google_search=types.GoogleSearch()),
            ]
            
            gen_config = types.GenerateContentConfig(
                tools=tools,
                response_mime_type="text/plain",
                temperature=0.2,
                max_output_tokens=4000
            )
            
            # Send request to the model using the client directly
            logging.info(f"Sending search request to Gemini for session {state.session_id}...")
            
            # Create contents with the specific instruction to use Google Search
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=search_prompt),
                    ],
                ),
            ]
            
            gen_response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-preview-04-17",
                contents=contents,
                config=gen_config
            )
            
            # Extract search results
            search_data = gen_response.text
            logging.info(f"Received search results for session {state.session_id} ({len(search_data)} chars)")
            
            # If search data is too short, likely no proper search was performed
            if len(search_data) < 500:
                logging.warning(f"Search results suspiciously short ({len(search_data)} chars). May not have used Google Search.")
                # InterviewAgent prompt: Fallback search prompt with explicit search queries
                retry_prompt = f"""
                YOU MUST USE THE GOOGLE SEARCH TOOL for this task. Run the following searches:
                
                1. "{company} company information"
                2. "{company} {job_title} job requirements"
                3. "{company} recent news"
                4. "{job_title} interview questions"
                
                For each search, provide the results you find, summarized in a clear format.
                Do not proceed without using the Google Search tool.
                """
                
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=retry_prompt),
                        ],
                    ),
                ]
                
                logging.info(f"Retrying search with more explicit instructions for session {state.session_id}")
                retry_response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash-preview-04-17",
                    contents=contents,
                    config=gen_config
                )
                
                # Use the retry response if it's longer
                retry_data = retry_response.text
                if len(retry_data) > len(search_data):
                    search_data = retry_data
                    logging.info(f"Using retry search results with length {len(search_data)} chars")
            
            # Store the search data in the session state
            state.search_state.search_data = search_data
            
            # If we have grounding metadata, we can save sources too
            if gen_response.candidates and hasattr(gen_response.candidates[0], 'grounding_metadata') and gen_response.candidates[0].grounding_metadata:
                gm = gen_response.candidates[0].grounding_metadata
                
                if hasattr(gm, 'web_search_queries') and gm.web_search_queries:
                    state.search_state.web_search_queries = list(gm.web_search_queries)
                    logging.info(f"Saved {len(state.search_state.web_search_queries)} web search queries")
                    
                if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                    sources = []
                    for chunk in gm.grounding_chunks:
                        if hasattr(chunk, 'web') and chunk.web:
                            uri = getattr(chunk.web, 'uri', 'N/A')
                            title = getattr(chunk.web, 'title', 'N/A')
                            sources.append({"title": title, "uri": uri})
                    state.search_state.grounding_sources = sources
                    logging.info(f"Saved {len(sources)} grounding sources")
                    
            return "Search completed successfully."
            
        except Exception as e:
            logging.error(f"Error in search: {e}", exc_info=True)
            state.search_state.search_data = "Error: Could not retrieve interview preparation data."
            return "I encountered an error while searching for information. Let's continue with the interview."

    async def _ask_next_question(self, state: SessionState) -> str:
        """Generate the next interview question based on search data and conversation history."""
        interview = state.interview_state
        search_data = state.search_state.search_data
        
        paragraphs = self._split_into_paragraphs(search_data)
        if not paragraphs:
             return "I seem to be having trouble with my source material. Let's try something else. What is the proudest achievement in your career so far?"

        difficulty_levels = {0: "Easy", 1: "Medium", 2: "Hard"}
        
        previous_score_was_zero = interview.scores and interview.scores[-1] == 0
        if previous_score_was_zero:
            logging.info("Previous answer was incorrect (score: 0), adjusting difficulty.")

        question_counter = interview.paragraph_question_counter
        current_index = interview.current_paragraph_index % len(paragraphs)
        
        if interview.questions and interview.answers:
            if not previous_score_was_zero:
                question_counter += 1
                interview.paragraph_question_counter = question_counter
            
            if question_counter >= 3:
                interview.current_paragraph_index = (current_index + 1) % len(paragraphs)
                interview.paragraph_question_counter = 0
                logging.info(f"Asked 3 questions, moving to paragraph {interview.current_paragraph_index}")
        
        current_index = interview.current_paragraph_index % len(paragraphs)
        base_question_counter = interview.paragraph_question_counter
        
        adjusted_counter = base_question_counter - 1 if previous_score_was_zero and base_question_counter > 0 else base_question_counter
        difficulty = difficulty_levels.get(adjusted_counter, "Medium")
        
        # --- New: Define Question Styles based on Difficulty ---
        question_styles = {
            "Easy": {
                "title": "Foundational Check",
                "instruction": "Ask a direct question to verify fundamental knowledge of a key technology, concept, or process mentioned in the context. Example: 'What is the primary purpose of [Technology X] in the context of [Domain Y]?'"
            },
            "Medium": {
                "title": "Scenario-Based Problem",
                "instruction": "Create a brief, realistic scenario based on the context. The user should be the protagonist. Ask them how they would approach or solve this specific problem. Frame it like: 'Imagine you've been tasked with improving [Metric A] for [Product B]. Given the context, what would be your initial steps?'"
            },
            "Hard": {
                "title": "Complex Design Challenge",
                "instruction": "Pose a complex, multi-faceted challenge that requires a high-level design or strategic plan. This should involve trade-offs and justification. Frame it like: 'You are leading the project to integrate [System A] with [System B]. Based on the context, outline your proposed architecture. What are the key risks and how would you mitigate them?'"
            }
        }
        selected_style = question_styles[difficulty]

        context = paragraphs[current_index]
        
        conversation_context = ""
        if interview.questions and interview.answers:
            last_question = interview.questions[-1]
            last_answer = interview.answers[-1]
            conversation_context = f"Previous Question: {last_question}\nUser's Last Answer: {last_answer}"
        
        job_title = state.profiling_state.user_data.get('job_title', 'the role')
        company = state.profiling_state.user_data.get('target_company', 'the company')

        prompt = f"""
        You are an expert technical interviewer for {company}, preparing to interview a candidate for a {job_title} position. Your task is to generate one insightful, scenario-based interview question.

        **CANDIDATE PROFILE:**
        - Role: {job_title}
        - Target Company: {company}

        **INTERVIEW CONTEXT (Obtained from internal research documents):**
        ---
        {context}
        ---

        **RECENT CONVERSATION (for context):**
        {conversation_context}

        **YOUR TASK:**
        Generate ONE interview question following the specific style guide below.

        - **Difficulty Level:** {difficulty}
        - **Question Style:** {selected_style['title']}
        - **Instruction for this style:** {selected_style['instruction']}

        **CRITICAL RULES:**
        1.  **Create a Narrative:** Immerse the candidate in a scenario. Use phrases like "Imagine you are the {job_title} at {company}..." or "Suppose we need to...".
        2.  **Be Specific:** Directly reference technologies, projects, or concepts from the **INTERVIEW CONTEXT**. Do not ask generic questions.
        3.  **Test, Don't Just Ask:** The question should test knowledge, problem-solving, or design skills, not just opinions.
        4.  **Natural Tone:** Sound like a real interviewer, not a bot reading a script. Do not mention "based on the context" in your question.
        5.  **Output ONLY the question.** No pre-amble, no explanations, just the single question itself.
        """
        
        try:
            response = await self._call_llm(prompt, is_json=False)
            next_question = response.strip()
            interview.questions.append(next_question)
            return next_question
        except Exception as e:
            logging.error(f"Error generating next question: {e}", exc_info=True)
            return "I'm having trouble formulating my next question. Could we take a brief pause?"

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Helper method to split search data into meaningful paragraphs."""
        if not text:
            return []
        
        # Split by double newlines to get paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out very short paragraphs and those without meaningful content
        substantial_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 250 and any(char.isalpha() for char in p)]
        
        # If we don't have enough substantial paragraphs, fall back to lower threshold
        if len(substantial_paragraphs) < 3:
            substantial_paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 100 and any(char.isalpha() for char in p)]
        
        # Skip the first paragraph if it's just an introduction or summary and we have other paragraphs
        if len(substantial_paragraphs) > 1 and len(substantial_paragraphs[0]) < 300 and "provides" in substantial_paragraphs[0].lower():
            substantial_paragraphs = substantial_paragraphs[1:]
        
        return substantial_paragraphs

    async def _evaluate_answer(self, state: SessionState) -> dict:
        """Evaluate the most recent answer."""
        interview = state.interview_state
        
        if not interview.questions or len(interview.questions) == 0 or len(interview.answers) == 0:
            return {"evaluation": "No answer to evaluate"}
        
        # Get the most recent question and answer
        question = interview.questions[-1]
        answer = interview.answers[-1]
        
        # Get the search data and split into paragraphs
        search_data = state.search_state.search_data
        paragraphs = self._split_into_paragraphs(search_data)
        
        # Use the same paragraph that was used to generate the question
        current_index = interview.current_paragraph_index % len(paragraphs)
        context_paragraph = paragraphs[current_index]
        
        # Determine difficulty level based on question counter
        base_question_counter = interview.paragraph_question_counter
        difficulty_mapping = {0: "Easy", 1: "Medium", 2: "Hard"}
        difficulty = difficulty_mapping.get(base_question_counter, "Medium")
        max_score_mapping = {"Easy": 1, "Medium": 2, "Hard": 3}
        max_score = max_score_mapping.get(difficulty, 2)
        
        # InterviewAgent prompt: Evaluates candidate answers against reference information
        prompt = f"""
        You are an AI interview evaluator. Evaluate the candidate's answer to this interview question.
        
        QUESTION: {question}
        
        CANDIDATE'S ANSWER: {answer}
        
        REFERENCE INFORMATION:
        {context_paragraph}
        
        DIFFICULTY LEVEL: {difficulty} (Max Score: {max_score})
        
        Provide a detailed evaluation that:
        1. Assesses the accuracy of the answer relative to the reference information
        2. Evaluates the completeness and depth of the answer
        3. Considers the clarity of communication
        4. Assigns a score out of {max_score} (can include decimals)
        
        IMPORTANT: A score of 0 should be used for completely incorrect or significantly misleading answers.
        Scores greater than 0 indicate varying levels of correctness.
        
        Return a JSON object with this structure:
        {{
            "evaluation": "A concise evaluation of the answer (1-2 sentences)",
            "feedback": "Specific feedback on strengths and areas for improvement",
            "score": The numerical score (between 0 and {max_score}),
            "strengths": ["Strength 1", "Strength 2", ...], 
            "weaknesses": ["Weakness 1", "Weakness 2", ...],
            "improvement_tips": ["Tip 1", "Tip 2", ...],
            "scoring_breakdown": {{
                "accuracy": score between 0-100,
                "completeness": score between 0-100,
                "clarity": score between 0-100
            }},
            "is_correct": true/false (assign false if score is 0, true otherwise)
        }}
        """
        
        try:
            # Create appropriate content structure
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation parameters
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
            
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            # Process response
            result_text = response.text.strip()
            cleaned_text = re.sub(r"```json\s*|\s*```", "", result_text)
            
            try:
                result = json.loads(cleaned_text)
                
                # Ensure score is treated as numeric
                if "score" in result:
                    result["score"] = float(result.get("score", 0))
                    
                # Add is_correct flag if not already present
                if "is_correct" not in result:
                    result["is_correct"] = result.get("score", 0) > 0
                    
                # Add metadata to the evaluation result
                result["question"] = question
                result["answer"] = answer
                result["difficulty"] = difficulty
                result["max_score"] = max_score
                result["timestamp"] = datetime.now(timezone.utc).isoformat()
                
                # Create and store a DetailedEvaluation object
                question_id = f"Q{len(interview.questions)}"  # This creates Q1, Q2, Q3, etc. - 1-based indexing
                detailed_eval = {
                    "question_id": question_id,
                    "score": result.get("score", 0),
                    "strengths": result.get("strengths", []),
                    "weaknesses": result.get("weaknesses", []),
                    "improvement_tips": result.get("improvement_tips", []),
                    "scoring_breakdown": result.get("scoring_breakdown", result.get("rubric_section", {}))
                }
                
                # Import at function level to avoid circular imports
                from models import DetailedEvaluation
                
                detailed_evaluation = DetailedEvaluation(**detailed_eval)
                state.detailed_evaluations.append(detailed_evaluation)
                
                # Log evaluation result
                logging.info(f"Evaluation: Score={result.get('score', 0)}/{max_score}, Difficulty={difficulty}, Correct={result.get('is_correct', False)}")
                logging.info(f"Added detailed evaluation with ID: {question_id}, Total evaluations: {len(state.detailed_evaluations)}")
                
                return result
            except json.JSONDecodeError:
                logging.error(f"Failed to parse evaluation JSON: {cleaned_text}")
                default_result = {
                    "evaluation": "The answer could not be automatically evaluated.",
                    "feedback": "Please continue with the interview.",
                    "score": 1,
                    "question": question,
                    "answer": answer,
                    "difficulty": difficulty,
                    "max_score": max_score,
                    "is_correct": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "strengths": ["Unable to determine strengths due to evaluation error"],
                    "weaknesses": ["Unable to determine weaknesses due to evaluation error"],
                    "improvement_tips": ["Continue to the next question"],
                    "scoring_breakdown": {"accuracy": 50, "completeness": 50, "clarity": 50}
                }
                
                # Create and store a minimal DetailedEvaluation for the failure case
                question_id = f"Q{len(interview.questions)}"
                from models import DetailedEvaluation
                
                detailed_evaluation = DetailedEvaluation(
                    question_id=question_id,
                    score=1,
                    strengths=["Unable to determine strengths due to evaluation error"],
                    weaknesses=["Unable to determine weaknesses due to evaluation error"],
                    improvement_tips=["Continue to the next question"]
                )
                state.detailed_evaluations.append(detailed_evaluation)
                
                return default_result
        except Exception as e:
            logging.error(f"Error generating evaluation: {e}", exc_info=True)
            default_result = {
                "evaluation": "An error occurred during evaluation.",
                "feedback": "Please continue with the interview.",
                "score": 1,
                "question": question,
                "answer": answer,
                "difficulty": difficulty,
                "max_score": max_score,
                "is_correct": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "strengths": ["Unable to determine strengths due to error"],
                "weaknesses": ["Unable to determine weaknesses due to error"],
                "improvement_tips": ["Continue to the next question"],
                "scoring_breakdown": {"accuracy": 50, "completeness": 50, "clarity": 50}
            }
            
            # Create and store a minimal DetailedEvaluation for the error case
            question_id = f"Q{len(interview.questions)}"
            from models import DetailedEvaluation
            
            detailed_evaluation = DetailedEvaluation(
                question_id=question_id,
                score=1,
                strengths=["Unable to determine strengths due to error"],
                weaknesses=["Unable to determine weaknesses due to error"],
                improvement_tips=["Continue to the next question"]
            )
            state.detailed_evaluations.append(detailed_evaluation)
            
            return default_result

    async def _generate_summary(self, state: SessionState) -> str:
        """Generate a summary of the interview."""
        interview = state.interview_state
        profile = state.profiling_state.user_data
        
        # Format conversation for summary
        # Create pairs of questions and answers with scores
        qa_pairs = []
        for i in range(min(len(interview.questions), len(interview.answers))):
            score = interview.scores[i] if i < len(interview.scores) else None
            max_score = interview.max_scores[i] if i < len(interview.max_scores) else None
            
            qa_pair = {
                "question": interview.questions[i],
                "answer": interview.answers[i],
                "score": score,
                "max_score": max_score
            }
            qa_pairs.append(qa_pair)
        
        # Calculate interview statistics
        total_score = sum(interview.scores) if interview.scores else 0
        total_max = sum(interview.max_scores) if interview.max_scores else 0
        percentage = (total_score / total_max * 100) if total_max > 0 else 0
        
        job_title = profile.get('job_title', 'the role')
        company = profile.get('target_company', 'the company')
        
        # InterviewAgent prompt: Generates comprehensive interview performance summary
        prompt = f"""
        Generate a comprehensive interview summary for a candidate who interviewed for a {job_title} position at {company}.
        
        CANDIDATE PROFILE:
        {json.dumps(profile, indent=2)}
        
        INTERVIEW SUMMARY:
        - Questions asked: {len(interview.questions)}
        - Total score: {total_score:.1f}/{total_max} ({percentage:.1f}%)
        
        QUESTION-ANSWER PAIRS:
        {json.dumps(qa_pairs, indent=2)}
        
        Based on the above interview, generate a detailed summary that includes:
        1. Overall performance assessment
        2. Key strengths demonstrated
        3. Areas for improvement
        4. Technical skills assessment
        5. Fit for the {job_title} role at {company}
        
        Format the summary as a structured JSON with these sections:
        - Summary: Overall performance summary (1-2 paragraphs)
        - Skills: Technical skills assessment (with string ratings like Excellent, Good, Average, Poor)
        - Strengths: Key strengths demonstrated
        - AreasToImprove: Suggested areas for growth
        - FitForRole: Assessment of fit for the position
        - AdditionalDetails: Any other relevant observations
        
        Keep the response strictly in JSON format.
        """
        
        try:
            # Create appropriate content structure
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation parameters
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
            
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            # Process the response
            result_text = response.text.strip()
            cleaned_text = re.sub(r"```json\s*|\s*```", "", result_text)
            
            try:
                summary_data = json.loads(cleaned_text)
                
                # Store the summary in the session state
                state.summary_data = summary_data
                
                return json.dumps(summary_data, indent=2)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse summary JSON: {cleaned_text}")
                summary_data = {
                    "Summary": "Failed to generate a proper summary. The interview was completed, but the summary generation encountered an error.",
                    "Error": "JSON parsing error"
                }
                state.summary_data = summary_data
                return json.dumps(summary_data, indent=2)
            
        except Exception as e:
            logging.error(f"Error generating summary: {e}", exc_info=True)
            summary_data = {
                "Summary": "An error occurred while generating the interview summary.",
                "Error": str(e)
            }
            state.summary_data = summary_data
            return json.dumps(summary_data, indent=2)

    async def _find_matches(self, state: SessionState) -> List[dict]:
        """Find potential referrer matches based on the interview summary."""
        try:
            logging.info(f"Finding matches for session {state.session_id}")
            
            # Get the summary from the state
            summary = state.summary_data
            if not summary:
                logging.warning(f"No summary available for session {state.session_id}")
                return []
            
            # Get MongoDB client from database module
            user_collection = get_user_collection()
            if user_collection is None:
                logging.error("MongoDB collection not available")
                return []
            
            # Get the MongoDB client from the collection
            mongo_client = user_collection.database.client
                
            # Connect to referrer_db for matching
            match_db = mongo_client["referrer_db"]
            referrers_collection = match_db["employee_embeddings"]
            logging.info(f"Connected to DB: {match_db.name}, Collection: {referrers_collection.name}")
            
            # Build candidate description from summary
            description_parts = []
            
            # Extract candidate name (for identification only, not for matching)
            candidate_name = ""
            summary_text = summary.get("Summary", "")
            if isinstance(summary_text, str) and "interviewed for" in summary_text:
                candidate_name = summary_text.split("interviewed for")[0].strip()
            if not candidate_name and isinstance(summary.get("Additionaldetails"), dict):
                candidate_name = summary.get("Additionaldetails", {}).get("Name", "")
            
            # Extract job title and company from profiling state
            profile = state.profiling_state.user_data
            job_title = profile.get("job_title", "")
            target_company = profile.get("target_company", "")
            
            # Build description using only relevant matching parameters (not name)
            description_parts.append(f"Seeking Job Title: {job_title}")
            description_parts.append(f"Targeting Company: {target_company}")
            
            # Extract skills
            if isinstance(summary.get("Skills"), dict) and isinstance(summary["Skills"].get("TechnicalSkills"), list):
                skill_names = [s.get("name") if isinstance(s, dict) else str(s) for s in summary["Skills"]["TechnicalSkills"] if s]
                skill_names = [s_name for s_name in skill_names if s_name]  # Filter out None/empty strings
                if skill_names: 
                    description_parts.append(f"Technical Skills: {', '.join(skill_names)}")
            
            if summary.get("Summary"): 
                description_parts.append(f"Overall Summary: {summary.get('Summary')}")
            
            candidate_description = "\n".join(filter(None, description_parts))
            
            if not candidate_description:
                logging.error(f"Could not construct candidate description for {state.session_id}")
                return []
            
            logging.info(f"Generated candidate description for matching: {candidate_description[:100]}...")
            
            # Generate embedding for the candidate description
            user_embedding = await self._generate_embedding(candidate_description)
            
            results = []
            
            # Try vector search if we have an embedding
            if user_embedding and referrers_collection is not None:
                logging.info(f"Searching vector matches in {referrers_collection.name}")
                try:
                    pipeline = [
                        {"$search": {
                            "index": "vector_index",  # Ensure this index exists
                            "knnBeta": {"vector": user_embedding, "path": "embedding", "k": 5}
                        }},
                        {"$project": {
                            "score": {"$meta": "searchScore"}, "_id": 1, "Name": 1, 
                            "profile": 1, "employeeId": 1
                        }},
                        {"$limit": 5}
                    ]
                    vector_results = list(referrers_collection.aggregate(pipeline))
                    logging.info(f"Found {len(vector_results)} matches via vector search.")
                    results.extend(vector_results)
                except Exception as e:
                    logging.error(f"Error during vector search: {str(e)}")
            
            # Fallback to text search if no results
            if not results and referrers_collection is not None:
                logging.info(f"Fallback to text search in {referrers_collection.name}")
                text_query = {}
                query_conditions = []
                
                if job_title:
                    query_conditions.append({"$or": [
                        {"profile.workExperience.jobTitle": {"$regex": job_title, "$options": "i"}},
                        {"profile.jobTitle": {"$regex": job_title, "$options": "i"}},
                        {"job_title": {"$regex": job_title, "$options": "i"}}
                    ]})
                
                if target_company:
                    query_conditions.append({"$or": [
                        {"profile.workExperience.company": {"$regex": target_company, "$options": "i"}},
                        {"profile.company": {"$regex": target_company, "$options": "i"}},
                        {"company": {"$regex": target_company, "$options": "i"}}
                    ]})
                
                if query_conditions:
                    text_query["$and"] = query_conditions
                
                if text_query:
                    logging.info(f"Text searching with query: {text_query}")
                    try:
                        text_results = list(referrers_collection.find(text_query).limit(5))
                        for r in text_results:
                            if "score" not in r: r["score"] = 0.5  # Placeholder
                        logging.info(f"Found {len(text_results)} matches via text search.")
                        
                        # Add only if not already found by vector search
                        existing_ids = {str(res.get("_id")) for res in results}
                        for r_text in text_results:
                            if str(r_text.get("_id")) not in existing_ids:
                                results.append(r_text)
                    except Exception as e:
                        logging.error(f"Error in text search fallback: {str(e)}")
            
            # Format the results
            formatted_matches = []
            for referrer_doc in results:
                try:
                    name_to_use = referrer_doc.get("Name")  # From vector $project
                    
                    # If 'profile' object is projected and contains more details
                    profile_data_nested = referrer_doc.get("profile", {}) if isinstance(referrer_doc.get("profile"), dict) else {}
                    
                    if not name_to_use and isinstance(profile_data_nested, dict):
                        name_to_use = profile_data_nested.get("Name")
                    if not name_to_use:  # Fallback
                        name_to_use = "Unknown Referrer"
                    
                    company = "Unknown Company"
                    role = "Unknown Role"
                    
                    work_experience = profile_data_nested.get("workExperience")
                    if isinstance(work_experience, dict):
                        company = work_experience.get("company", company)
                        role = work_experience.get("jobTitle", role)
                    else:
                        company = profile_data_nested.get("company", company)
                        role = profile_data_nested.get("jobTitle", role)
                    
                    # Try from referrer_doc directly if still unknown
                    if company == "Unknown Company": 
                        company = referrer_doc.get("company", "Unknown Company")
                    if role == "Unknown Role": 
                        role = referrer_doc.get("job_title", referrer_doc.get("jobTitle", "Unknown Role"))
                    
                    # Get the employee ID (prioritize the specific employeeId field)
                    employee_id = None
                    if "employeeId" in referrer_doc and referrer_doc["employeeId"]:
                        # Convert ObjectId to string if needed
                        employee_id = str(referrer_doc["employeeId"])
                    
                    # Convert ObjectId to string if needed
                    doc_id = referrer_doc.get("_id")
                    id_str = str(doc_id) if doc_id else f"match_{len(formatted_matches)}"
                    
                    # Ensure match_score is a number
                    score = referrer_doc.get("score", 0.6)
                    if not isinstance(score, (int, float)):
                        try:
                            score = float(score)
                        except:
                            score = 0.6
                            
                    # Create a complete match object
                    formatted_match = {
                        "id": id_str,
                        "employee_id": employee_id,
                        "name": name_to_use,
                        "company": company,
                        "role": role,
                        "match_score": score
                    }
                    
                    # Add detailed logging for this match
                    logging.info(f"Creating match: id={id_str}, name={name_to_use}, company={company}, role={role}, score={score}")
                    
                    formatted_matches.append(formatted_match)
                except Exception as e:
                    logging.error(f"Error formatting a match: {str(e)}")
                    # Don't add the problematic match
            
            # Ensure we have at least some results
            if not formatted_matches and results:
                logging.warning("No matches were successfully formatted despite having results. Adding debug match.")
                # Add a log message with the raw results for debugging
                try:
                    logging.info(f"Raw results sample: {json.dumps(results[0])}")
                except:
                    logging.info("Could not serialize raw results for logging")
                    
            matches_count = len(formatted_matches)
            logging.info(f"Formatted {matches_count} matches from {len(results)} results")
            
            # Ensure we don't return more than 5 matches
            if matches_count > 5:
                logging.info(f"Limiting matches from {matches_count} to 5")
                formatted_matches = formatted_matches[:5]
            
            return formatted_matches
            
        except Exception as e:
            logging.error(f"Error in _find_matches: {str(e)}", exc_info=True)
            return []
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the input text."""
        try:
            import google.generativeai as genai
            
            # Clean and prepare text
            text_to_embed = text.replace("\n", " ")
            logging.info(f"Generating embedding for text: '{text_to_embed[:50]}...'")
            
            # Initialize the client with API key
            api_key = os.getenv("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
            
            # Use the embedding model
            embedding_model_name = "models/text-embedding-004"
            
            # Generate the embedding
            result = await asyncio.to_thread(
                genai.embed_content,
                model=embedding_model_name,
                content=text_to_embed,
                task_type="RETRIEVAL_QUERY"
            )
            
            # Extract the embedding values from the response
            if result and 'embedding' in result:
                logging.info("Embedding generated successfully.")
                return result['embedding']
            else:
                logging.error(f"Error generating embedding: No 'embedding' field in API response")
                return None
        except Exception as e:
            logging.error(f"Error in generate_embedding: {str(e)}", exc_info=True)
            return None


# Instantiate stateless agents
profiling_agent = ProfilingAgent()
interview_agent = InterviewAgent()

class DoubtAgent:
    """
    Handles post-interview questions about evaluations, scores, and performance.
    """
    def __init__(self):
        # Check if environment variable is set
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Initialize client
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.5-flash-preview-04-17"

    async def _call_llm(self, prompt: str, is_json: bool = False) -> str:
        """Helper method to call the LLM with proper settings."""
        try:
            # Create appropriate content structure
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                    ],
                ),
            ]
            
            # Configure generation parameters
            gen_config = types.GenerateContentConfig(
                response_mime_type="application/json" if is_json else "text/plain",
                temperature=0.2,
            )
            
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            return response.text.strip()
        except Exception as e:
            logging.error(f"LLM call failed: {str(e)}", exc_info=True)
            if "rate limit" in str(e).lower():
                return '{"error": "API rate limit exceeded. Please try again in a moment."}' if is_json else "I'm currently experiencing high demand. Could you try again in a moment?"
            return '{"error": "Failed to get a valid response from the model."}' if is_json else "I'm having a little trouble processing that. Could we try again?"

    async def process(self, user_input: str, state: SessionState) -> str:
        """Process user questions about their interview and evaluation."""
        # Ensure we have evaluations to reference
        if not state.detailed_evaluations:
            logging.warning(f"No detailed evaluations found in state for doubt processing")
            return "I don't have any evaluation data to reference. Let's discuss something else."
        
        # Log evaluation data for debugging
        logging.info(f"Processing doubt query. Available evaluations: {len(state.detailed_evaluations)}")
        logging.info(f"Questions asked: {len(state.interview_state.questions)}")
        for i, q in enumerate(state.interview_state.questions):
            logging.info(f"Q{i+1}: {q[:50]}...")
        
        # Determine what the user is asking about
        context = await self._get_question_context(user_input, state)
        logging.info(f"Question context determined: {context}")
        
        # If asking about a specific question/answer
        if context.get("question_id"):
            return await self._handle_question_about_answer(context["question_id"], state, user_input)
        
        # For general performance questions
        if context.get("is_general_performance"):
            return await self._handle_general_question(user_input, state)
        
        # For questions about matches
        if context.get("is_about_matches"):
            return await self._handle_match_question(user_input, state)
        
        # Handle out-of-scope questions
        return (
            "I can answer questions about your interview performance, "
            "specific answers you gave, or your potential matches. "
            "For example, you can ask 'What did I do well in my second answer?' "
            "or 'Why did I score low on the technical skills question?'"
        )

    async def _get_question_context(self, user_input: str, state: SessionState) -> dict:
        """Extract which question/answer the user is asking about."""
        question_count = len(state.interview_state.questions)
        
        # Create a list of all questions with their IDs for context
        questions_with_ids = []
        for i, q in enumerate(state.interview_state.questions):
            q_id = f"Q{i+1}"  # Use 1-based indexing for human readability
            questions_with_ids.append({"id": q_id, "question": q})
        
        logging.info(f"Context extraction - Questions available: {question_count}")
        logging.info(f"Context extraction - User input: '{user_input}'")
        
        # DoubtAgent prompt: Analyzes user questions to determine what they're asking about
        prompt = f"""
        Analyze this user question: "{user_input}"
        
        The user has completed an interview with {question_count} questions.
        The questions were:
        {json.dumps(questions_with_ids)}
        
        Determine:
        1. Is the user asking about a specific question/answer? If so, which one (Q1, Q2, Q3, etc.)?
        2. Is this a general performance question about overall scores or skills?
        3. Is this about matching or referrals?
        
        Return a JSON with:
        {{
            "question_id": "Q1" or "Q2" or null if not specific,
            "is_general_performance": true/false,
            "is_about_matches": true/false,
            "query_type": "specific_answer", "general_performance", "matches", or "out_of_scope"
        }}
        """
        
        # Call LLM for classification
        response = await self._call_llm(prompt, is_json=True)
        try:
            cleaned_text = re.sub(r"```json\s*|\s*```", "", response)
            context = json.loads(cleaned_text)
            logging.info(f"Context extraction result: {context}")
            return context
        except json.JSONDecodeError:
            logging.error(f"Failed to decode context JSON: {response}")
            return {"question_id": None, "is_general_performance": False, "is_about_matches": False, "query_type": "out_of_scope"}

    async def _handle_question_about_answer(self, question_id: str, state: SessionState, user_input: str) -> str:
        """Handle specific questions about individual answers."""
        # Add debugging
        logging.info(f"Handling question about answer: {question_id}")
        logging.info(f"Available detailed_evaluations: {[e.question_id for e in state.detailed_evaluations]}")
        logging.info(f"Questions: {len(state.interview_state.questions)} / Answers: {len(state.interview_state.answers)}")
        
        # Find the evaluation for this question
        matching_eval = None
        for eval in state.detailed_evaluations:
            if eval.question_id == question_id:
                matching_eval = eval
                logging.info(f"Found matching evaluation for {question_id}")
                break
        
        if not matching_eval:
            logging.warning(f"No evaluation found for {question_id}")
            return f"I don't have detailed evaluation data for question {question_id.replace('Q', '')}. Let's discuss something else."
        
        # Get the actual question and answer
        q_index = int(question_id.replace('Q', '')) - 1
        if q_index < 0 or q_index >= len(state.interview_state.questions) or q_index >= len(state.interview_state.answers):
            logging.warning(f"Invalid question index {q_index} for {question_id}")
            return f"I couldn't find the details for question {question_id.replace('Q', '')}."
            
        question = state.interview_state.questions[q_index]
        answer = state.interview_state.answers[q_index]
        score = state.interview_state.scores[q_index] if q_index < len(state.interview_state.scores) else "Not scored"
        max_score = state.interview_state.max_scores[q_index] if q_index < len(state.interview_state.max_scores) else "Unknown"
        
        # DoubtAgent prompt: Generates detailed explanations about specific answer evaluations
        prompt = f"""
        The user is asking: "{user_input}"
        
        About this interview Q&A:
        Question: {question}
        User's answer: {answer}
        Score: {score}/{max_score}
        
        Evaluation details:
        Strengths: {json.dumps(matching_eval.strengths)}
        Weaknesses: {json.dumps(matching_eval.weaknesses)}
        Improvement tips: {json.dumps(matching_eval.improvement_tips)}
        Scoring breakdown: {json.dumps(matching_eval.scoring_breakdown) if matching_eval.scoring_breakdown else "Not available"}
        
        IMPORTANT INSTRUCTIONS:
        1. Explain mark deductions concisely:
           - Identify specific missed points
           - State exact deduction reasons
           - Highlight incomplete/incorrect parts
           - Compare with expected answer
        
        2. Address weaknesses directly:
           - Explain errors concisely
           - Provide improvement suggestions
           - Use examples from their answer
        
        3. For partial scores, specify exactly what cost them points
        
        4. Reference scoring components (accuracy, completeness, clarity)
        
        5. Be concise but thorough
        
        Provide a direct, specific response to the user's question.
        Be honest about weaknesses while remaining empathetic.
        """
        
        logging.info(f"Generated prompt for question {question_id}")
        
        # Call the LLM for a personalized response
        return await self._call_llm(prompt)

    async def _handle_general_question(self, user_input: str, state: SessionState) -> str:
        """Handle general questions about interview performance."""
        # Get aggregate performance data
        total_score = sum(state.interview_state.scores) if state.interview_state.scores else 0
        total_max = sum(state.interview_state.max_scores) if state.interview_state.max_scores else 0
        percentage = (total_score / total_max * 100) if total_max > 0 else 0
        
        # Get summary data if available
        summary = state.summary_data or {}
        
        # Collect all strengths and weaknesses from detailed evaluations
        all_strengths = []
        all_weaknesses = []
        question_scores = []
        
        for i, eval in enumerate(state.detailed_evaluations):
            q_num = i + 1
            score = state.interview_state.scores[i] if i < len(state.interview_state.scores) else 0
            max_score = state.interview_state.max_scores[i] if i < len(state.interview_state.max_scores) else 1
            question_scores.append({
                "question_id": f"Q{q_num}",
                "score": score,
                "max_score": max_score,
                "percentage": (score / max_score * 100) if max_score > 0 else 0
            })
            all_strengths.extend(eval.strengths)
            all_weaknesses.extend(eval.weaknesses)
        
        # DoubtAgent prompt: Analyzes overall interview performance and provides detailed feedback
        prompt = f"""
        The user is asking: "{user_input}"
        
        About their overall interview performance:
        - Questions asked: {len(state.interview_state.questions)}
        - Total score: {total_score:.1f}/{total_max} ({percentage:.1f}%)
        
        Individual question scores:
        {json.dumps(question_scores)}
        
        Key strengths identified across all answers:
        {json.dumps(all_strengths)}
        
        Key weaknesses identified across all answers:
        {json.dumps(all_weaknesses)}
        
        Summary data:
        {json.dumps(summary)}
        
        IMPORTANT INSTRUCTIONS:
        1. Provide a concise, focused analysis of the user's performance
        
        2. Clearly identify their main weaknesses:
           - Name the lowest-scoring questions
           - Highlight 2-3 key weakness patterns
           - Specify the most critical knowledge gaps
        
        3. Structure your response efficiently:
           - First: 1-2 sentences on main improvement areas
           - Then: 1 sentence acknowledging strengths
        
        4. For overall performance questions:
           - Give specific scores for their weakest areas
           - Mention 1-2 concrete improvement suggestions
           - Be direct about how they compare to expectations
        
        5. For score-specific questions:
           - Name the 1-2 questions that most affected their score
           - Briefly explain the specific issues in those answers
        
        Generate a brief, direct response (3-5 sentences) that answers the user's question with specific data points.
        Be clear about improvement areas while remaining constructive.
        """
        
        # Call the LLM for a personalized response
        return await self._call_llm(prompt)

    async def _handle_match_question(self, user_input: str, state: SessionState) -> str:
        """Handle questions about matches/referrals."""
        # Check if we have matches
        if not state.matches:
            return "I don't currently have any match data for potential referrals. Let me check if there's an issue with your session."
        
        # DoubtAgent prompt: Explains potential referral matches to the user
        prompt = f"""
        The user is asking: "{user_input}"
        
        About their potential matches for referrals:
        {json.dumps(state.matches)}
        
        Generate a helpful, conversational response that addresses the user's question about their matches.
        Explain why these matches were selected if that information is available.
        Keep your response concise (3-5 sentences) but informative.
        """
        
        # Call the LLM for a personalized response
        return await self._call_llm(prompt)

# Add instantiation for the new agent
doubt_agent = DoubtAgent() 
