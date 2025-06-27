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
        self.fields = ["name", "target_company", "job_title", "domain", "user_experience", "user_long_term_objective"]
        self.field_prompts = {
            "name": "Ask for the user's full name in a friendly, professional tone.",
            "target_company": "Ask for the name of the company the user is interested in (focus on ONE specific company).",
            "job_title": "Ask for the specific job title the user is seeking (e.g., 'AI Engineer', 'Product Manager').",
            "domain": "Ask the user to select your domain of interest from the following options: Finance, Operations, HR, Marketing, IT.",
            "user_experience": "Ask for a brief summary of the user's relevant work experience or background.",
            "user_long_term_objective": "Ask about the user's long-term career objectives or aspirations."
        }
        # Check if environment variable is set
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Initialize client instead of GenerativeModel
        self.client = genai.Client(api_key=api_key)
        # Primary model
        self.model_name = "gemini-2.5-flash-preview-04-17"
        # Fallback model for when primary hits rate limits
        self.fallback_model_name = "gemini-2.5-flash-lite-preview-06-17"

    async def _call_llm(self, prompt: str, is_json: bool = False) -> str:
        """Helper method to call the LLM with proper settings."""
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
            
        # First try with primary model
        try:
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            return response.text.strip()
        except Exception as primary_error:
            error_str = str(primary_error).lower()
            if ("rate limit" in error_str or "resource_exhausted" in error_str or "quota" in error_str):
                # Log fallback in orange
                logger.info(f"\033[33mPrimary model hit quota/rate limit, falling back to {self.fallback_model_name}\033[0m")
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.fallback_model_name,
                        contents=contents,
                        config=gen_config
                    )
                    logger.info(f"\033[33mSuccessfully used fallback model {self.fallback_model_name}\033[0m")
                    return response.text.strip()
                except Exception as fallback_error:
                    logging.error(f"Fallback model also failed: {str(fallback_error)}", exc_info=True)
                    return '{"error": "All available models failed to respond."}' if is_json else "I'm currently experiencing technical difficulties. Could we try again in a moment?"
            else:
                # For non-rate-limit errors
                logging.error(f"LLM call failed with primary model: {str(primary_error)}", exc_info=True)
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

        2.  **Extract Multiple Fields**: IMPORTANT - The user may provide information for multiple fields at once. For example, they might say "I'm John Smith looking for a Data Scientist role at Amazon". This contains name, job_title, and target_company.
            *   Check if the response contains information for other fields beyond the one you're currently asking about.
            *   If it does, extract that information too and include it in the JSON response under "additional_fields".

        3.  **Act Based on Intent**:
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
          "ai_response": "The helpful, user-facing response you crafted.",
          "additional_fields": {{
            "field_name1": "extracted_value1",
            "field_name2": "extracted_value2"
          }}
        }}

        The "additional_fields" should only be included if you detected information for other fields beyond the one currently being asked about. Valid field names are: "name", "target_company", "job_title", "domain", "user_experience", "user_long_term_objective".

        **Example for this request:**
        If the user says "I'm John looking for a Data Scientist role at Amazon" when asked about their name, your output should be:
        {{
            "intent": "answering",
            "is_valid": true,
            "extracted_value": "John",
            "ai_response": "Got it, John! What specific department or domain are you interested in at Amazon? For example, Machine Learning, Data Engineering, Business Intelligence, etc.",
            "additional_fields": {{
                "job_title": "Data Scientist",
                "target_company": "Amazon"
            }}
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
                
                # Process any additional fields that were extracted
                additional_fields = validation_result.get("additional_fields", {})
                if additional_fields:
                    logger.info(f"Found additional fields in response: {json.dumps(additional_fields)}")
                    
                    # Store additional field values and track which fields were filled
                    filled_fields = []
                    for field_name, field_value in additional_fields.items():
                        if field_name in self.fields and field_name != current_field:
                            profiling.user_data[field_name] = field_value
                            filled_fields.append(field_name)
                            logger.info(f"Stored additional field: {field_name} = {field_value}")
                
                # Determine the next field to ask about
                next_field_index = current_field_index + 1
                
                # Skip fields that were already filled from additional_fields
                while (next_field_index < len(self.fields) and 
                       self.fields[next_field_index] in profiling.user_data and 
                       profiling.user_data[self.fields[next_field_index]]):
                    logger.info(f"Skipping field {self.fields[next_field_index]} as it was already filled")
                    next_field_index += 1
                
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
    MAX_QUESTIONS = 10  # Updated from 3 to 10 to ensure all questions are asked before finalizing

    def __init__(self):
        self.fields = ["name", "target_company", "job_title", "domain", "user_experience", "user_long_term_objective"]
        self.field_prompts = {
            "name": "Ask for the user's full name in a friendly, professional tone.",
            "target_company": "Ask for the name of the company the user is interested in (focus on ONE specific company).",
            "job_title": "Ask for the specific job title the user is seeking (e.g., 'AI Engineer', 'Product Manager').",
            "domain": "Ask about their area of expertise or department of interest (e.g., 'Machine Learning', 'Product Management', 'Data Science', 'Software Engineering', 'Marketing', 'Finance').",
            "user_experience": "Ask for a brief summary of the user's relevant work experience or background.",
            "user_long_term_objective": "Ask about the user's long-term career objectives or aspirations."
        }
        # Check if environment variable is set
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        # Initialize client instead of GenerativeModel
        self.client = genai.Client(api_key=api_key)
        # Primary model
        self.model_name = "gemini-2.5-flash-preview-04-17"
        # Fallback model for when primary hits rate limits
        self.fallback_model_name = "gemini-2.5-flash-lite-preview-06-17"

    async def _call_llm(self, prompt: str, is_json: bool = False) -> str:
        """Helper method to call the LLM with proper settings."""
        contents = [
            types.Content(
                role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
            ),
        ]
        
        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json" if is_json else "text/plain",
            temperature=0,
        )
            
        # First try with primary model
        try:
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            return response.text.strip()
        except Exception as primary_error:
            error_str = str(primary_error).lower()
            if ("rate limit" in error_str or "resource_exhausted" in error_str or "quota" in error_str):
                # Log fallback in orange
                logger.info(f"\033[33mPrimary model hit quota/rate limit, falling back to {self.fallback_model_name}\033[0m")
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.fallback_model_name,
                        contents=contents,
                        config=gen_config
                    )
                    logger.info(f"\033[33mSuccessfully used fallback model {self.fallback_model_name}\033[0m")
                    return response.text.strip()
                except Exception as fallback_error:
                    logging.error(f"Fallback model also failed: {str(fallback_error)}", exc_info=True)
                    return '{"error": "All available models failed to respond."}' if is_json else "I'm currently experiencing technical difficulties. Could we try again in a moment?"
            else:
                # For non-rate-limit errors
                logging.error(f"LLM call failed with primary model: {str(primary_error)}", exc_info=True)
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
            
        # If we have a previous question, treat the user input as an answer
        if interview.questions:
            # Check if we already have the same number of answers as questions
            # If so, this means we've already processed this answer
            if len(interview.answers) < len(interview.questions):
                # Only append the answer if it hasn't been added yet
                interview.answers.append(user_input)
                
                # Evaluate the answer
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
        total_questions_asked = interview.domain_questions_asked + interview.technical_questions_asked + interview.soft_skill_questions_asked
        logger.info(f"Question count: domain={interview.domain_questions_asked}, technical={interview.technical_questions_asked}, soft_skill={interview.soft_skill_questions_asked}, total={total_questions_asked}/{self.MAX_QUESTIONS}")
        if total_questions_asked >= self.MAX_QUESTIONS:
            # Generate summary and transition to summary stage
            state.current_stage = 'summary'
            await self._generate_summary(state)
            state.current_stage = 'complete'
            final_message = "Thank you for completing the interview! I'll now analyze your responses and prepare a summary of your performance along with potential matches."
            interview.conversation_history.append({"assistant": final_message})
            
            # Find matches based on summary and return them to the user
            matches = await self._find_matches(state)
            
            # Format matches for display to user in Markdown format
            if matches:
                match_count = len(matches)
                match_details = "\n\n## Potential Referrers Who Match Your Profile\n"
                for i, match in enumerate(matches[:5], 1):  # Show up to 5 matches
                    match_name = match.get("name", "Unnamed Referrer")
                    match_company = match.get("company", "Unknown Company")
                    match_role = match.get("role", "Unknown Role")
                    match_score = match.get("match_score", 0.0)
                    score_percentage = int(match_score * 100) if isinstance(match_score, (int, float)) else "--"
                    
                    match_details += f"\n### {i}. {match_name}\n"
                    match_details += f"- **Company:** {match_company}\n"
                    match_details += f"- **Role:** {match_role}\n"
                    match_details += f"- **Match Score:** {score_percentage}%\n"
                
                return f"## Interview Completed! 🎉\n\nThank you for completing the interview process. Based on your responses, I've found **{match_count} potential referrals** that match your profile.{match_details}\n\n> You can now reach out to these professionals for referral opportunities. Feel free to ask any questions about your results."
            else:
                return "## Interview Completed! 🎉\n\nThank you for completing the interview process! I've analyzed your responses and am currently searching for potential referrals that match your profile. Please check back shortly for your matches."
        
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
        domain = profile.get('domain', 'general')
        
        # InterviewAgent prompt: Uses Google Search to gather company and job role information
        search_prompt = f"""
        You are a Search Agent with access to the Google Search API. Your task is to gather detailed information about the company '{company}'
        and the job role '{job_title}' in the domain/department of '{domain}'. 
        
        IMPORTANT: YOU MUST USE THE GOOGLE SEARCH TOOL to find recent and accurate information. DO NOT make up information.
        
        For each category below, run at least 1-2 specific Google searches and include the information you found:
        
        CATEGORY 1 - DOMAIN-SPECIFIC INFORMATION:
        1. Business problems and challenges faced by {company} in the {domain} department/domain
        2. Recent projects or initiatives in the {domain} area at {company}
        3. How {company}'s {domain} department operates and its role in the company
        4. Specific {domain} strategies or approaches used at {company}
        
        CATEGORY 2 - TECHNICAL INFORMATION:
        1. Technical skills and qualifications typically required for {job_title} at {company}
        2. Technologies, tools, and systems used by {company} for {job_title} roles
        3. Technical challenges and problems solved by {job_title} professionals at {company}
        4. Industry-standard technical practices relevant to {job_title} at {company}
        
        CATEGORY 3 - SOFT SKILLS & CULTURE:
        1. Company culture and work environment at {company}
        2. Soft skills valued for {job_title} positions at {company}
        3. Team structure and collaboration approaches at {company}
        4. Leadership style and management philosophy at {company}
        5. Interview process and typical questions for a {job_title} role at {company}
        
        For each search query, paste the actual search query you use and then summarize the information found.
        
        Structure your response in three clearly separated sections with these exact headings:
        
        [DOMAIN-SPECIFIC]
        (Include all domain-specific information here)
        
        [TECHNICAL]
        (Include all technical information here)
        
        [SOFT-SKILLS]
        (Include all soft skills and culture information here)
        
        Include source URLs where relevant in each section.
        
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
                
                DOMAIN-SPECIFIC SEARCHES:
                1. "{company} {domain} department challenges"
                2. "{company} {domain} projects"
                
                TECHNICAL SEARCHES:
                1. "{company} {job_title} technical skills"
                2. "{company} {job_title} technical interview questions"
                
                SOFT-SKILLS SEARCHES:
                1. "{company} company culture"
                2. "{company} {job_title} soft skills"
                3. "{company} interview process for {job_title}"
                
                For each, provide a detailed summary of the information found.
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
            
            # Store the full search data in the session state
            state.search_state.search_data = search_data
            
            # Parse and categorize the search data
            try:
                # Extract domain-specific data
                domain_match = re.search(r'\[DOMAIN-SPECIFIC\](.*?)(?=\[TECHNICAL\]|\[SOFT-SKILLS\]|$)', search_data, re.DOTALL)
                if domain_match:
                    state.search_state.domain_specific_data = domain_match.group(1).strip()
                    logging.info(f"Extracted domain-specific data: {len(state.search_state.domain_specific_data)} chars")
                
                # Extract technical data
                technical_match = re.search(r'\[TECHNICAL\](.*?)(?=\[DOMAIN-SPECIFIC\]|\[SOFT-SKILLS\]|$)', search_data, re.DOTALL)
                if technical_match:
                    state.search_state.technical_data = technical_match.group(1).strip()
                    logging.info(f"Extracted technical data: {len(state.search_state.technical_data)} chars")
                
                # Extract soft skills data
                soft_skills_match = re.search(r'\[SOFT-SKILLS\](.*?)(?=\[DOMAIN-SPECIFIC\]|\[TECHNICAL\]|$)', search_data, re.DOTALL)
                if soft_skills_match:
                    state.search_state.soft_skills_data = soft_skills_match.group(1).strip()
                    logging.info(f"Extracted soft skills data: {len(state.search_state.soft_skills_data)} chars")
                
                # If any category is missing, use a portion of the full search data as fallback
                if not state.search_state.domain_specific_data or not state.search_state.technical_data or not state.search_state.soft_skills_data:
                    logging.warning("Some categories missing from search results, using fallback approach")
                    paragraphs = self._split_into_paragraphs(search_data)
                    
                    if paragraphs:
                        third = max(1, len(paragraphs) // 3)
                        if not state.search_state.domain_specific_data:
                            state.search_state.domain_specific_data = "\n\n".join(paragraphs[:third])
                        if not state.search_state.technical_data:
                            state.search_state.technical_data = "\n\n".join(paragraphs[third:2*third])
                        if not state.search_state.soft_skills_data:
                            state.search_state.soft_skills_data = "\n\n".join(paragraphs[2*third:])
            except Exception as e:
                logging.error(f"Error categorizing search data: {e}", exc_info=True)
                # Use the full search data for all categories as fallback
                paragraphs = self._split_into_paragraphs(search_data)
                if paragraphs:
                    third = max(1, len(paragraphs) // 3)
                    state.search_state.domain_specific_data = "\n\n".join(paragraphs[:third])
                    state.search_state.technical_data = "\n\n".join(paragraphs[third:2*third])
                    state.search_state.soft_skills_data = "\n\n".join(paragraphs[2*third:])
            
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
        """Generate the next interview question based on current interview stage and search data."""
        interview = state.interview_state
        total_questions_asked = (
            interview.domain_questions_asked +
            interview.technical_questions_asked +
            interview.soft_skill_questions_asked
        )
        if total_questions_asked >= self.MAX_QUESTIONS:
            logger.info(f"Global max questions ({self.MAX_QUESTIONS}) reached, finalizing interview.")
            asyncio.create_task(self.finalize_interview(state))
            return "That concludes our interview questions. Thank you for your responses. I'll now analyze your answers and provide feedback."

        # Determine which type of question to ask based on the current interview stage
        if interview.current_stage == "domain_specific":
            # Check if we've reached the limit for domain-specific questions
            if interview.domain_questions_asked >= interview.domain_questions_limit:
                # Transition to technical questions
                interview.current_stage = "technical"
                logger.info(f"Transitioning from domain_specific to technical questions for session {state.session_id}")
                return await self._ask_next_question(state)
            else:
                # Continue with domain-specific questions
                next_question = await self._generate_domain_question(state)
                interview.domain_questions_asked += 1
                interview.question_categories.append("domain_specific")
                logger.info(f"Generated domain-specific question ({interview.domain_questions_asked}/{interview.domain_questions_limit}) for session {state.session_id}")
                return next_question
                
        elif interview.current_stage == "technical":
            # Check if we've reached the limit for technical questions
            if interview.technical_questions_asked >= interview.technical_questions_limit:
                # Transition to soft skill questions
                interview.current_stage = "soft_skill"
                logger.info(f"Transitioning from technical to soft_skill questions for session {state.session_id}")
                return await self._ask_next_question(state)
            else:
                # Continue with technical questions
                next_question = await self._generate_technical_question(state)
                interview.technical_questions_asked += 1
                interview.question_categories.append("technical")
                logger.info(f"Generated technical question ({interview.technical_questions_asked}/{interview.technical_questions_limit}) for session {state.session_id}")
                return next_question
                
        elif interview.current_stage == "soft_skill":
            # Check if we've reached the limit for soft skill questions
            if interview.soft_skill_questions_asked >= interview.soft_skill_questions_limit:
                # Interview is complete
                logger.info(f"All questions completed for session {state.session_id}, finalizing interview")
                asyncio.create_task(self.finalize_interview(state))
                return "That concludes our interview questions. Thank you for your responses. I'll now analyze your answers and provide feedback."
            else:
                # Continue with soft skill questions
                next_question = await self._generate_soft_skill_question(state)
                interview.soft_skill_questions_asked += 1
                interview.question_categories.append("soft_skill")
                logger.info(f"Generated soft skill question ({interview.soft_skill_questions_asked}/{interview.soft_skill_questions_limit}) for session {state.session_id}")
                return next_question
        
        # Fallback in case of unexpected state
        logger.warning(f"Unexpected interview stage: {interview.current_stage} for session {state.session_id}")
        return "I seem to be having trouble with my interview flow. Let me ask you: What do you consider your greatest professional achievement so far?"

    async def _generate_domain_question(self, state: SessionState) -> str:
        """Generate a domain-specific question based on the search data."""
        interview = state.interview_state
        
        # Use domain-specific data instead of general search data
        domain_data = state.search_state.domain_specific_data
        
        # Fallback to general search data if domain-specific data is not available
        if not domain_data:
            domain_data = state.search_state.search_data
            logging.warning("Domain-specific data not available, falling back to general search data")
        
        paragraphs = self._split_into_paragraphs(domain_data)
        if not paragraphs:
             return "I seem to be having trouble with my source material. Let's try something else. What is the proudest achievement in your career so far?"

        # Calculate difficulty based on total questions asked across all categories
        total_questions_asked = interview.domain_questions_asked + interview.technical_questions_asked + interview.soft_skill_questions_asked
        
        # Cycle through difficulty levels: Easy (0,3,6,9), Medium (1,4,7), Hard (2,5,8)
        difficulty_index = total_questions_asked % 3
        difficulty_levels = {0: "Easy", 1: "Medium", 2: "Hard"}
        difficulty = difficulty_levels[difficulty_index]
        
        logger.info(f"Generating domain question #{interview.domain_questions_asked+1} with difficulty: {difficulty} (based on total questions: {total_questions_asked})")
        
        # Select paragraph based on question count
        current_index = interview.domain_questions_asked % len(paragraphs)
        context = paragraphs[current_index]
        logger.info(f"Domain Q uses paragraph index: {current_index}")
        
        # --- Domain Question Styles based on Difficulty ---
        question_styles = {
            "Easy": {
                "title": "Foundational Check",
                "instruction": "Ask a direct question to verify fundamental knowledge of a key technology, concept, or process mentioned in the context. Example: 'What is the primary purpose of [Technology X] in the context of [Domain Y]?'"
            },
            "Medium": {
                "title": "Scenario-Based Problem",
                "instruction": "Create a brief, realistic scenario based on the context. Frame it like: 'As the [role] at [company], how would you address [specific challenge]?' or 'What approach would you take to solve [specific problem] in our [product/system]?'"
            },
            "Hard": {
                "title": "Complex Design Challenge",
                "instruction": "Pose a complex, multi-faceted challenge that requires a high-level design or strategic plan. Frame it like: 'Our team needs to [complex task]. What architecture would you propose and what key considerations would guide your approach?'"
            }
        }
        selected_style = question_styles[difficulty]
        
        conversation_context = ""
        if interview.questions and interview.answers:
            last_question = interview.questions[-1]
            last_answer = interview.answers[-1]
            conversation_context = f"Previous Question: {last_question}\nUser's Last Answer: {last_answer}"
        
        job_title = state.profiling_state.user_data.get('job_title', 'the role')
        company = state.profiling_state.user_data.get('target_company', 'the company')
        domain = state.profiling_state.user_data.get('domain', 'general')

        prompt = f"""
        You are an expert interviewer for {company}, preparing to interview a candidate for a {job_title} position in the {domain} department/domain. Your task is to generate one insightful, scenario-based interview question that focuses on domain-specific business problems.

        **CANDIDATE PROFILE:**
        - Role: {job_title}
        - Target Company: {company}
        - Department/Domain: {domain}

        **DOMAIN-SPECIFIC CONTEXT (Obtained from internal research documents):**
        ---
        {context}
        ---

        **RECENT CONVERSATION (for context):**
        {conversation_context}

        **PREVIOUSLY ASKED QUESTIONS:**
        {interview.asked_question_texts}

        **YOUR TASK:**
        Generate ONE domain-specific interview question following the specific style guide below. The question should focus on business problems and challenges specific to the {domain} department at {company}.

        - **Difficulty Level:** {difficulty}
        - **Question Style:** {selected_style['title']}
        - **Instruction for this style:** {selected_style['instruction']}

        **CRITICAL RULES:**
        1.  **Vary Question Formats:** Use different phrasings for each question. Avoid starting every question with the same words (like "imagine" or "suppose"). Instead, vary with formats like:
            - "How would you approach..."
            - "What strategy would you implement for..."
            - "Tell me about how you would solve..."
            - "In your role as {job_title} in the {domain} department, what would be your approach to..."
            - "Based on your experience, how would you handle..."
        2.  **Be Specific:** Directly reference business problems, projects, or concepts from the DOMAIN-SPECIFIC CONTEXT.
        3.  **Sound Natural:** Use the conversational tone of a real human interviewer, not an AI.
        4.  **Test Domain Knowledge:** The question should test domain-specific knowledge, problem-solving, or strategic thinking related to the {domain} department.
        5.  **Make it Concise:** Keep the question clear and to the point - avoid unnecessarily long scenarios.
        6.  **Output ONLY the question.** No pre-amble, no explanations, just the single question itself.
        7.  **Check Previous Questions:** Ensure your question is different in both content and phrasing from previously asked questions.
        8.  **Format in Markdown:** Format your question using Markdown syntax. Use appropriate formatting like:
            - Bold (**text**) for emphasis on key terms
            - Bullet points for lists if needed
        """
        
        try:
            response = await self._call_llm(prompt, is_json=False)
            next_question = response.strip()
            
            # Store the question to prevent repetition
            interview.asked_question_texts.append(next_question)
            interview.questions.append(next_question)
            return next_question
        except Exception as e:
            logging.error(f"Error generating domain question: {e}", exc_info=True)
            return "I'm having trouble formulating my next domain-specific question. Could you tell me about your experience with this type of role?"

    async def _generate_technical_question(self, state: SessionState) -> str:
        """Generate a technical question based on the search data."""
        
        interview = state.interview_state
        technical_data = state.search_state.technical_data
        soft_skills_data = state.search_state.soft_skills_data
        
        # Fallback to general search data if technical data is not available
        if not technical_data:
            technical_data = state.search_state.search_data
            logging.warning("Technical data not available, falling back to general search data")

        if not soft_skills_data:
            soft_skills_data = "No specific data on soft skills or interview process found."
            
        paragraphs = self._split_into_paragraphs(technical_data)
        
        if not paragraphs:
            return "Based on your experience, could you describe a complex technical challenge you've faced and how you solved it?"

        # Determine difficulty based on question progression
        difficulty = "easy"
        total_questions_asked = interview.domain_questions_asked + interview.technical_questions_asked + interview.soft_skill_questions_asked
        if total_questions_asked > (self.MAX_QUESTIONS / 2):
            difficulty = "hard"
        elif total_questions_asked > (self.MAX_QUESTIONS / 4):
            difficulty = "medium"
            
        logging.info(f"Generating technical question #{interview.technical_questions_asked+1} with difficulty: {difficulty} (based on total questions: {total_questions_asked})")

        # Cycle through paragraphs
        current_index = interview.technical_questions_asked % len(paragraphs)
        context_paragraph = paragraphs[current_index]
        logger.info(f"Technical Q uses paragraph index: {current_index}")
        
        profile = state.profiling_state.user_data
        job_title = profile.get('job_title', 'the specified role')
        company = profile.get('target_company', 'the specified company')
        domain = profile.get('domain', 'general')
        experience = profile.get('user_experience', 'Not provided')
        objective = profile.get('user_long_term_objective', 'Not provided')

        prompt = f"""
        You are an expert interviewer for {company}, preparing to interview a candidate for a {job_title} position in the {domain} department/domain.
        Your task is to generate one insightful, scenario-based interview question that focuses on technical skills.

        **CANDIDATE PROFILE:**
        - Job Title: {job_title}
        - Department/Domain: {domain}
        - Experience Summary: {experience}
        - Career Goal: {objective}

        **TECHNICAL CONTEXT (Obtained from internal research documents):**
        {technical_data}

        **INTERVIEW PROCESS CONTEXT (Obtained from internal research documents):**
        {soft_skills_data}

        **GUIDELINES:**
        1.  **Generate ONE question** based on the provided context.
        2.  **Be Specific:** Reference technologies, tools, or problems from the TECHNICAL CONTEXT.
        3.  **Scenario-Based:** Frame the question as a hypothetical problem or scenario.
        4.  **Test Technical Skills:** The question should test problem-solving, design, or coding skills.
        5.  **Difficulty Level: {difficulty}**. Adjust the complexity of the question accordingly.
        6.  **If the context includes details about the company's interview process, try to frame your question in a way that aligns with it (e.g., if they favor whiteboarding, ask a question suitable for that format).**

        **Do not ask for definitions.** Ask how the candidate would apply concepts.
        
        Your response should contain ONLY the question text, without any preamble, conversational filler, or explanation.
        """
        
        return await self._call_llm(prompt)

    async def _generate_soft_skill_question(self, state: SessionState) -> str:
        """Generate a soft-skill question based on the search data."""
        
        interview = state.interview_state
        soft_skills_data = state.search_state.soft_skills_data
        
        # Fallback to general search data if soft skills data is not available
        if not soft_skills_data:
            soft_skills_data = state.search_state.search_data
            logging.warning("Soft skills data not available, falling back to general search data")

        paragraphs = self._split_into_paragraphs(soft_skills_data)
        
        if not paragraphs:
            return "Can you tell me about a time you had a disagreement with a coworker and how you resolved it?"

        # Determine difficulty based on question progression
        difficulty = "easy"
        total_questions_asked = interview.domain_questions_asked + interview.technical_questions_asked + interview.soft_skill_questions_asked
        if total_questions_asked > (self.MAX_QUESTIONS / 2):
            difficulty = "hard"
        elif total_questions_asked > (self.MAX_QUESTIONS / 4):
            difficulty = "medium"
            
        logging.info(f"Generating soft skill question #{interview.soft_skill_questions_asked+1} with difficulty: {difficulty} (based on total questions: {total_questions_asked})")

        # Cycle through paragraphs
        current_index = interview.soft_skill_questions_asked % len(paragraphs)
        context_paragraph = paragraphs[current_index]
        logger.info(f"Soft skill Q uses paragraph index: {current_index}")
        
        profile = state.profiling_state.user_data
        job_title = profile.get('job_title', 'the specified role')
        company = profile.get('target_company', 'the specified company')
        domain = profile.get('domain', 'general')
        experience = profile.get('user_experience', 'Not provided')
        objective = profile.get('user_long_term_objective', 'Not provided')

        prompt = f"""
        You are an expert interviewer for {company}, preparing to interview a candidate for a {job_title} position in the {domain} department/domain. 
        Your task is to generate one insightful, behavioral interview question that focuses on soft skills and cultural fit.
        
        **CANDIDATE PROFILE:**
        - Job Title: {job_title}
        - Department/Domain: {domain}
        - Experience Summary: {experience}
        - Career Goal: {objective}

        **SOFT SKILLS & CULTURE CONTEXT (Obtained from internal research documents):**
        {soft_skills_data}
        
        **GUIDELINES:**
        1.  **Generate ONE question** based on the provided context.
        2.  **Behavioral Focus:** Ask for specific examples from the candidate's past experiences. Instead of always using "Tell me about a time when...", vary your phrasing. Use alternatives like:
            - "Can you walk me through a situation where..."
            - "Describe a challenging project you were a part of. What was your specific role and how did you handle..."
            - "How would you approach a situation where..."
            - "Give me an example of a time you had to influence a team decision. What was your approach?"
        3.  **Align with Culture:** The question should relate to the values, team structure, or work environment described in the SOFT SKILLS & CULTURE CONTEXT.
        4.  **Difficulty Level: {difficulty}**. Adjust the complexity of the scenario accordingly.
        5.  **If the context includes details about the company's interview process, try to frame your question in a way that aligns with it.**

        Your response should contain ONLY the question text, without any preamble, conversational filler, or explanation.
        """
        
        return await self._call_llm(prompt)

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
        
        # Calculate difficulty based on the current question's position
        # We use the length of answers (which should now be equal to the current question index)
        # This gives us the 0-indexed position of the current question being evaluated
        question_position = len(interview.answers) - 1
        
        # Cycle through difficulty levels: Easy (0,3,6,9), Medium (1,4,7), Hard (2,5,8)
        difficulty_index = question_position % 3
        difficulty_mapping = {0: "Easy", 1: "Medium", 2: "Hard"}
        difficulty = difficulty_mapping.get(difficulty_index, "Medium")
        max_score_mapping = {"Easy": 1, "Medium": 2, "Hard": 3}
        max_score = max_score_mapping.get(difficulty, 2)
        
        logger.info(f"Evaluating answer for question #{question_position+1} with difficulty: {difficulty}, max score: {max_score}")
        
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
        # Primary model
        self.model_name = "gemini-2.5-flash-preview-04-17"
        # Fallback model for when primary hits rate limits
        self.fallback_model_name = "gemini-2.5-flash-lite-preview-06-17"

    async def _call_llm(self, prompt: str, is_json: bool = False) -> str:
        """Helper method to call the LLM with proper settings."""
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
            
        # First try with primary model
        try:
            # Call the model using client
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=gen_config
            )
            
            return response.text.strip()
        except Exception as primary_error:
            error_str = str(primary_error).lower()
            if ("rate limit" in error_str or "resource_exhausted" in error_str or "quota" in error_str):
                # Log fallback in orange
                logger.info(f"\033[33mPrimary model hit quota/rate limit, falling back to {self.fallback_model_name}\033[0m")
                try:
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.fallback_model_name,
                        contents=contents,
                        config=gen_config
                    )
                    logger.info(f"\033[33mSuccessfully used fallback model {self.fallback_model_name}\033[0m")
                    return response.text.strip()
                except Exception as fallback_error:
                    logging.error(f"Fallback model also failed: {str(fallback_error)}", exc_info=True)
                    return '{"error": "All available models failed to respond."}' if is_json else "I'm currently experiencing technical difficulties. Could we try again in a moment?"
            else:
                # For non-rate-limit errors
                logging.error(f"LLM call failed with primary model: {str(primary_error)}", exc_info=True)
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
        
        # DoubtAgent prompt: Generates concise explanations about specific answer evaluations with examples
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
        
        CRITICAL INSTRUCTIONS:
        1. Provide a detailed response that explains what went wrong and how to improve, but keep it concise.
        2. Focus ONLY on the user's question.
        3. Use bullet points for clarity.
        4. Format in Markdown.
        5. For each weakness or area of improvement:
           - Explain WHY it's a weakness (1-2 sentences max)
           - Provide ONE specific example of a better answer.
           - If possible, show a BEFORE/AFTER comparison (keep it brief).
        6. Limit your response to a maximum of 6 bullet points or 120 words.
        7. Make your explanations actionable and clear, but do not write more than necessary.
        8. If the user is asking about a technical question, include a code example only if it adds real value.
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
        
        # DoubtAgent prompt: Analyzes overall interview performance and provides detailed feedback with examples in Markdown
        prompt = f"""
        The user is asking: "{user_input}"
        
        About their overall interview performance:
        - Questions asked: {len(state.interview_state.questions)}
        - Total score: {total_score:.1f}/{total_max} ({percentage:.1f}%)
        
        Individual question scores:
        {json.dumps(question_scores)}
        
        Top strengths (from all answers):
        {json.dumps(all_strengths[:5])}
        
        Top weaknesses (from all answers):
        {json.dumps(all_weaknesses[:5])}
        
        CRITICAL INSTRUCTIONS:
        1. Provide a detailed response that explains what went wrong and how to improve, but keep it concise.
        2. Focus ONLY on the user's question.
        3. Use bullet points for clarity.
        4. Format in Markdown.
        5. For each weakness or area of improvement:
           - Explain WHY it's a weakness (1-2 sentences max)
           - Provide ONE specific example of a better answer.
           - If possible, show a BEFORE/AFTER comparison (keep it brief).
        6. Limit your response to a maximum of 6 bullet points or 120 words.
        7. Make your explanations actionable and clear, but do not write more than necessary.
        8. If discussing technical skills, include a code example only if it adds real value.
        """
        
        # Call the LLM for a personalized response
        return await self._call_llm(prompt)

    async def _handle_match_question(self, user_input: str, state: SessionState) -> str:
        """Handle questions about matches/referrals."""
        # Check if we have matches
        if not state.matches:
            return "## No Matches Found\n\nI don't currently have any match data for potential referrals. Let me check if there's an issue with your session."
        
        # Get detailed match information
        match_details = []
        for i, match in enumerate(state.matches[:5], 1):
            match_name = match.get("name", "Unnamed Referrer")
            match_company = match.get("company", "Unknown Company")
            match_role = match.get("role", "Unknown Role")
            match_score = match.get("match_score", 0.0)
            score_percentage = int(match_score * 100) if isinstance(match_score, (int, float)) else "--"
            
            match_details.append({
                "name": match_name,
                "company": match_company,
                "role": match_role,
                "score": score_percentage
            })
        
        # DoubtAgent prompt: Answers questions about potential referral matches concisely
        prompt = f"""
        The user is asking: "{user_input}"
        
        About their potential referral matches:
        {json.dumps(match_details)}
        
        CRITICAL INSTRUCTIONS:
        1. Provide an EXTREMELY CONCISE response
        2. Focus ONLY on the user's question
        3. Use bullet points for clarity
        4. Format in Markdown
        5. Be direct and to the point - no fluff or unnecessary explanations
        6. Answer ONLY what the user is specifically asking about
        

        """
        
        # Call the LLM for a personalized response
        return await self._call_llm(prompt)

# Add instantiation for the new agent
doubt_agent = DoubtAgent() 