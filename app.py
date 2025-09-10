from flask import Flask, request, Response, stream_with_context
from query_agent import get_agent_executor
from flask_cors import CORS
from langchain_core.messages import AIMessage
import json


def create_app():
    app = Flask(__name__)

    app.agent_executor = get_agent_executor()

    return app


app = create_app()
CORS(app)

@app.route("/chat")
def chat():
    agent_executor = app.agent_executor
    question = request.args.get("question")
    thread_id = request.args.get("thread_id")
    events = agent_executor.stream(
        {"messages": [("user", question)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values"
    )

    @stream_with_context
    def generate():
        for event in events:
            # Debug print to console
            for message in event["messages"]:
                if isinstance(message, AIMessage):
                    print((message.content or (message.additional_kwargs or {}).get("tool_calls", {})[0].get("function", {}).get("name")))
                    # Yield JSON as string
                    yield json.dumps({"message": (message.content or (message.additional_kwargs or {}).get("tool_calls", {})[0].get("function", {}).get("name"))}) + "\n"

    return Response(generate(), mimetype="application/json")
