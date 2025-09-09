# Flask Chat Backend

This is a simple **Flask backend app** that exposes a single endpoint:

### Endpoint
- `Get /chat`  
  Accepts two string parameters:
  - `question` → the user’s query  
  - `thread_id` → identifier for the conversation thread  

The endpoint connects to a **custom AI agent** that:
- Understands natural language questions
- Generates SQL queries with database context
- Fetches and returns results as responses

### Running the App
```bash
flask --app app run