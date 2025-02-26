from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.schema import SystemMessage,HumanMessage
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Set your Google Gemini API Key (Ensure it's set in environment variables)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Google Gemini API Key is missing. Set it as an environment variable.")

# Initialize LangChain with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Define request schema
class PromptRequest(BaseModel):
    vague_prompt: str
    purpose: str
    tone: str
    complexity: str  # Keep it as str if frontend sends it as a string
    target_audience: str
    format: str
    length: str
    keywords: list[str]  # ✅ Ensure this is a list
    style: str
    references: list[str]  # ✅ Ensure this is a list
    vocabulary_level: str
    emotion: str
    cultural_considerations: str
    visual_aids: list[str]  # ✅ FIXED: Change from str to list[str]
    interactivity: list[str]  # ✅ FIXED: Change from str to list[str]
    restrictions: str

# Define route for processing prompt
@app.get("/")
def landing():
    return "Welcome to landing page"

@app.post("/generate_prompt")
def generate_prompt(request: PromptRequest):
    try:
        # Construct the detailed prompt
        detailed_prompt = f'''
        [User's Vague Prompt]: "{request.vague_prompt}"
        
        [Purpose]: {request.purpose}
        [Tone]: {request.tone}
        [Complexity]: {request.complexity}
        [Target Audience]: {request.target_audience}
        [Format]: {request.format}
        [Length]: {request.length}
        [Keywords]: {', '.join(request.keywords)}
        [Style]: {request.style}
        [References]: {' | '.join(request.references)}
        [Vocabulary Level]: {request.vocabulary_level}
        [Emotion/Intent]: {request.emotion}
        [Cultural Considerations]: {request.cultural_considerations}
        [Visual Aids]: {request.visual_aids}
        [Interactivity]: {request.interactivity}
        [Restrictions]: {request.restrictions}
        
        Generate the most optimal and detailed prompt based on these parameters.
        Generated prompt must be less than 150 words long'''

        # Use LangChain with Gemini API
        messages = [
            ("system","You are an expert prompt engineer. Your task is to generate the best possible prompt."),
            ("user",detailed_prompt)
        ]
        response = llm.invoke(detailed_prompt)
        
        return {"optimized_prompt": response.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
