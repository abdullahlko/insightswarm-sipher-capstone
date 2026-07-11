from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_async_session
from typing import Any

from app.models.schemas import ResearchRequest, ResearchRunResponse
from app.models.models import ResearchRun

from app.graphs.research_graph import research_graph, ResearchState

from app.db.database import async_session_maker

router = APIRouter()

# --- Background Task Function ---
# We use a separate database session maker here because the request-scoped 
# session closes before the background task finishes.

async def run_research_background(initial_state: ResearchState):
    async with async_session_maker() as session:
        try:
            print(f"Background task: Starting graph execution for run: {initial_state['run_id']}")
            final_state: dict[str, Any] = await research_graph.ainvoke(initial_state)
            print(f"Background task: Graph execution completed. Final state report length: {len(final_state['final_report'])}")
            
            # Update the database with the final state
            run = await session.get(ResearchRun, initial_state['run_id'])
            if run:
                run.status = "completed"
                run.progress = 100
                await session.commit()
        except Exception as e:
            print(f"Background task: Graph execution error: {e}")
            run = await session.get(ResearchRun, initial_state['run_id'])
            if run:
                run.status = "failed"
                run.progress = 0
                run.error_message = str(e)
                await session.commit()

# --- API Endpoint to Start Research ---

@router.post("/research", response_model=ResearchRunResponse, status_code=201)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks, # Allows us to run the research graph in the background
    session: AsyncSession = Depends(get_async_session)
):
    # 1. Map the incoming Pydantic request to our SQLAlchemy model
    new_run = ResearchRun(
        topic=request.topic,
        instructions=request.instructions,
        depth=request.depth,
        status = "running"
    )
    
    # 2. Add to session and commit to the database asynchronously
    session.add(new_run)
    try:
        await session.commit()
        await session.refresh(new_run) # Refresh to get the generated ID and timestamps
    except Exception as e:
        await session.rollback()
        print(f"Database error: {e}") # For local debugging
        raise HTTPException(status_code=500, detail="Failed to save research run.")
    
    # 3. Prepare the initial state for LangGraph
    initial_state: ResearchState = {
        "run_id": str(new_run.id),
        "topic": new_run.topic,
        "instructions": new_run.instructions or "",
        "depth": new_run.depth,
        "sub_questions": [],
        "sources": [],
        "draft": "",
        "is_verified": False,
        "final_report": "",
        "error": ""
    }

    # 4. Schedule the background task to run the research graph
    background_tasks.add_task(run_research_background, initial_state)
    
    # 5. Return the new research run details to the client
    return new_run

# --- API Endpoint to Check Research Status ---

@router.get("/research/{run_id}", response_model=ResearchRunResponse)
async def get_research_status(
    run_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    run = await session.get(ResearchRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Research run not found.")
    return run