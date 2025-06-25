from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime, timezone

# --- Models for User Input & Basic Chat ---

class UserRequest(BaseModel):
    """Generic request model for endpoints that need a session and user message."""
    user_message: str
    session_id: str

class DetailedEvaluation(BaseModel):
    """Represents a detailed evaluation of a user's answer with granular feedback."""
    question_id: str  # Unique identifier for the question
    score: float      # Numeric score assigned to the answer
    strengths: List[str] = Field(default_factory=list)  # Specific strong points in the answer
    weaknesses: List[str] = Field(default_factory=list)  # Areas needing improvement
    improvement_tips: List[str] = Field(default_factory=list)  # Actionable suggestions
    scoring_breakdown: Optional[Dict[str, Any]] = None  # Structured scoring breakdown
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatRequest(BaseModel):
    """Request model specifically for the main /user/chat endpoint."""
    user_input: str
    session_id: str

class ChatResponse(BaseModel):
    """Response model for the main /user/chat endpoint."""
    session_id: str
    message: str

# --- Models for Initial Profiling Agent ---

class UserData(BaseModel):
    """Model for the user's profile data collected by the conversational agent."""
    name: str
    target_company: str
    target_role: str
    career_goals: str
    background: str

# --- Models for Search and Interview Phase ---

class GroundingSource(BaseModel):
    """Represents a single web source found by the search agent."""
    title: Optional[str] = None
    uri: Optional[str] = None
    
    # Allow for arbitrary extra fields to handle varying response structures from Google API
    model_config = {
        "extra": "allow",
        "validate_assignment": False,
        "populate_by_name": True
    }

class QuestionData(BaseModel):
    """Represents a single interview question with its associated metadata."""
    question_id: str
    text: str
    category: Literal["domain_specific", "technical", "soft_skill"] = "domain_specific"
    difficulty: Optional[str] = None
    source_paragraph: Optional[str] = None
    source_heading: Optional[str] = None

class DataSourceResponse(BaseModel):
    """Response from the initial search_agent call."""
    search_data: str
    web_search_queries: Optional[List[str]] = None
    grounding_sources: Optional[List[GroundingSource]] = None
    session_id: str

class QuestionResponse(BaseModel):
    """
    Represents the data returned when a new question is asked.
    It can also carry the evaluation of the *previous* answer.
    """
    question: str
    session_id: str
    category: Optional[str] = None  # Category of the question: domain_specific, technical, or soft_skill
    source_paragraph: Optional[str] = None
    source_heading: Optional[str] = None
    difficulty: Optional[str] = None  # Difficulty of the *new* question being asked
    # Evaluation of the *previous* answer
    evaluation: Optional[str] = None
    evaluation_reason: Optional[str] = None
    score: Optional[float] = None
    max_score: Optional[int] = None
    cumulative_score: Optional[float] = None
    total_max_score: Optional[int] = None
    score_percentage: Optional[float] = None

class EvaluationResponse(BaseModel):
    """Represents the direct output of the evaluation agent for a single answer."""
    evaluation: str
    reason: Optional[str] = None
    session_id: str
    score: Optional[float] = None
    max_score: Optional[int] = None
    cumulative_score: Optional[float] = None
    total_max_score: Optional[int] = None
    score_percentage: Optional[float] = None
    difficulty: Optional[str] = None # Difficulty of the question that was evaluated

# --- Models for Summary and Matching Phase ---

class SummaryRequest(BaseModel):
    """Request to generate an interview summary."""
    session_id: str

class SummaryResponse(BaseModel):
    """Response containing the generated interview summary."""
    summary: Dict[str, Any]
    session_id: str

class SummaryRetrievalResponse(BaseModel):
    """Response for the endpoint that retrieves a summary's status."""
    session_id: str
    summary: Optional[Dict[str, Any]] = None
    status: str  # e.g., "ready", "processing", "not_started", "error"
    message: str

class MatchingRequest(BaseModel):
    """Request to find referrers matching the user's profile."""
    session_id: str
    job_criteria: Optional[Dict[str, Any]] = None

class MatchingResponse(BaseModel):
    """Response containing a list of potential referrer matches."""
    session_id: str
    matches: list

# --- Core Session State Models ---

class ProfilingState(BaseModel):
    """Holds the state for the initial user data collection conversation."""
    current_field_index: int = 0
    user_data: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    last_question: Optional[str] = None
    early_search_initiated: bool = False

class InterviewState(BaseModel):
    """Holds the state for the Q&A part of the interview."""
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)
    question_categories: List[str] = Field(default_factory=list)  # Tracks category for each question
    answers: List[str] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)
    max_scores: List[float] = Field(default_factory=list)
    current_paragraph_index: int = 0
    paragraph_question_counter: int = 0
    questions_asked_count: int = 0
    first_question_generated: bool = False
    
    # For tracking the structured interview flow
    current_stage: Literal["domain_specific", "technical", "soft_skill"] = "domain_specific"
    domain_questions_asked: int = 0
    technical_questions_asked: int = 0
    soft_skill_questions_asked: int = 0
    
    # Maximum number of questions per category
    domain_questions_limit: int = 4  # Verified: 4 domain-specific questions
    technical_questions_limit: int = 3  # Verified: 3 technical questions
    soft_skill_questions_limit: int = 3  # Verified: 3 soft skill questions
    
    # Track asked questions to prevent repetition
    asked_question_texts: List[str] = Field(default_factory=list)

class SearchState(BaseModel):
    """Holds the data gathered by the search agent."""
    search_data: Optional[str] = None
    domain_specific_data: Optional[str] = None
    technical_data: Optional[str] = None
    soft_skills_data: Optional[str] = None
    grounding_sources: List[GroundingSource] = Field(default_factory=list)
    web_search_queries: List[str] = Field(default_factory=list)
    search_failed: bool = False

class SessionState(BaseModel):
    """
    The main state object for a single user session.
    This will be cached in memory and persisted to MongoDB.
    """
    session_id: str
    current_stage: Literal['profiling', 'interviewing', 'summary', 'complete', 'doubts'] = 'profiling'
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    profiling_state: ProfilingState = Field(default_factory=ProfilingState)
    search_state: SearchState = Field(default_factory=SearchState)
    interview_state: InterviewState = Field(default_factory=InterviewState)

    summary_data: Optional[Dict[str, Any]] = None
    matches: List[Dict[str, Any]] = Field(default_factory=list)
    detailed_evaluations: List[DetailedEvaluation] = Field(default_factory=list)
    
    # Fields for doubt handling
    doubt_queries_count: int = 0
    doubt_queries_limit: int = 20  # Updated from 3 to 20 to allow more follow-up questions
    doubt_session_start: Optional[datetime] = None
    doubt_conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    
    # Field to store pending messages when search data isn't ready
    pending_messages: List[str] = Field(default_factory=list)

    def update_timestamp(self):
        self.last_updated = datetime.now(timezone.utc) 