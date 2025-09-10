from flask import Flask, request
from query_agent import get_agent_executor
from flask_cors import CORS


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
    final_event = None
    for event in events:
        event["messages"][-1].pretty_print()
        final_event = event

    return {"message": final_event["messages"][-1].content}
