
# os.environ["OPENAI_API_KEY"] = "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"
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from werkzeug.utils import secure_filename
import tempfile
import shutil
import json
from uuid import uuid4
from typing import List

import openai
import requests
import os

import config
import db
from dotenv import load_dotenv
from os.path import join, dirname

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = FastAPI()

# Mount the static folder for static files like CSS and JS
app.mount("/static", StaticFiles(directory="static"), name="static")
# Templates configuration
templates = Jinja2Templates(directory="templates")

database_file = "database.json"
database = db.load(database_file)
settings = config.load("settings.json")

# Custom function to find jobs
def custom_job_finder_state(query: str):
    url = "https://ai.joblab.ai/get_job_matches"
    query_params = {
        "query": query,
        "page": 1,
        "size": 5,
    }
    headers = {"accept": "application/json"}
    response = requests.post(url, params=query_params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        total_jobs = data["total"]
        job_matches_data = [
            {
                "job_id": job["job_id"],
                "job_title": job["job_title"],
                "job_company": job["job_company"],
                "job_location": job["job_location"],
                "job_description": job["job_description"],
            }
            for job in data["items"]
        ]
        return job_matches_data


class Chat(BaseModel):
    chat_id: str


class Message(BaseModel):
    chat_id: str
    message: str


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ... [Other routes like /new_chat, /load_chat, /conversations, /send_message]


@app.post("/new_chat", response_class=HTMLResponse)
def new_chat(request: Request):
    chat_id = str(uuid4())

    thread = openai.beta.threads.create()

    chat = {
        "id": chat_id,
        "thread_id": thread.id,
        "title": "JobGPT chat",
    }

    database["conversations"][chat_id] = chat
    db.save(database_file, database)

    return templates.TemplateResponse("chat_button.html", {"request": request, "chat": chat})


@app.get("/load_chat/{chat_id}", response_class=HTMLResponse)
def load_chat(request: Request, chat_id: str):
    thread_id = database["conversations"][chat_id]["thread_id"]

    messages = openai.beta.threads.messages.list(
        thread_id=thread_id,
        order="desc",
    )

    message_list = []

    for message in messages.data:
        message_list.append(
            {"role": message.role, "content": message.content[0].text.value}
        )

    message_list = reversed(message_list)

    return templates.TemplateResponse("messages.html", {"request": request, "messages": message_list, "chat_id": chat_id})


@app.get("/conversations", response_class=HTMLResponse)
def conversations(request: Request):
    chats = database["conversations"].values()
    return templates.TemplateResponse("conversations.html", {"request": request, "conversations": chats})


@app.post("/send_message", response_class=HTMLResponse)
async def send_message(request: Request, file = File(None), message = None):
    form_data = await request.form()
    form_data = dict(form_data)
    print(dict(form_data))
    chat_id = form_data["chat_id"]
    file_ids = []

    if file:
        temp_dir = tempfile.mkdtemp()

        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(temp_dir, filename)

            print(f"Saving to {file_path}")

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            uploaded_file = openai.files.create(
                file=openai.file_from_path(file_path),
                purpose="assistants",
            )

            file_ids.append(uploaded_file.id)
        finally:
            shutil.rmtree(temp_dir)

    user_message = {"role": "user", "content": form_data["message"]}

    chat = database["conversations"][chat_id]

    # Add the message after handling the run
    openai.beta.threads.messages.create(
        thread_id=chat["thread_id"],
        role=user_message["role"],
        content=user_message["content"],
        file_ids=file_ids,
    )

    return templates.TemplateResponse("user_message.html", {"request": request, "chat_id": chat_id, "message": user_message})


@app.get("/get_response/{chat_id}", response_class=HTMLResponse)
def get_response(request: Request, chat_id: str):
    chat = database["conversations"][chat_id]

    # Create a new run
    run = openai.beta.threads.runs.create(
        thread_id=chat["thread_id"],
        assistant_id=settings["assistant_id"],
    )

    # Store the run_id in the chat object and save it
    chat["run_id"] = run.id
    db.save(database_file, database)

    # Retrieve the current run to check its status
    current_run = openai.beta.threads.runs.retrieve(
        run_id=run.id, thread_id=chat["thread_id"]
    )

    # Wait for the run to not be in an active state
    while current_run.status in ["queued", "in_progress", "cancelling"]:
        current_run = openai.beta.threads.runs.retrieve(
            run_id=run.id, thread_id=chat["thread_id"]
        )

        if current_run.status == "requires_action":
            tools_output = []
            for tool_call in current_run.required_action.submit_tool_outputs.tool_calls:
                f = tool_call.function
                if f.name == "custom_job_finder":
                    query = json.loads(f.arguments)["query"]
                    tool_result = custom_job_finder_state(query)
                    tool_result_json = json.dumps(tool_result)
                    tools_output.append(
                        {
                            "tool_call_id": tool_call.id,
                            "output": tool_result_json,
                        }
                    )

            openai.beta.threads.runs.submit_tool_outputs(
                thread_id=chat["thread_id"],
                run_id=run.id,
                tool_outputs=tools_output,
            )

            # Re-check the run status after handling requires_action
            current_run = openai.beta.threads.runs.retrieve(
                run_id=run.id, thread_id=chat["thread_id"]
            )

    # Retrieve the latest message after the run is completed or not active
    messages = openai.beta.threads.messages.list(
        thread_id=chat["thread_id"],
        order="desc",
        limit=1,
    )

    assistant_message = {"role": "assistant", "content": messages.data[0].content[0].text.value}
    return templates.TemplateResponse("assistant_message.html", {"request": request, "message": assistant_message})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
