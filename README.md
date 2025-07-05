# QueryFlowAI
- Accepts a user query - Detects if the query is a **coding-related** question or a **simple/general** query
- Uses **LangGraph** to build a stateful flow graph to route and process the query
-  Leverages **Gemini (Google GenAI)** to generate structured or conversational responses based on the type of query

  ## 🔧 Tech Stack

| Technology       | Purpose                                  |
|------------------|-------------------------------------------|
| Python           | Core programming language                |
| [LangGraph](https://github.com/langchain-ai/langgraph)     | To define stateful, modular graph-based workflows |
| [LangChain](https://github.com/langchain-ai/langchain)     | Chat model abstraction and structured output       |
| [Google GenAI](https://ai.google.dev/)                     | LLM used for query classification and response     |
| Pydantic          | Input/output validation via `BaseModel` |
| Python Dotenv     | To manage API keys using `.env` files    |
| TypedDict         | For strict state management of the graph |


Set up your .env file
Create a .env file in the root directory and add your Google API key:
GOOGLE_API_KEY=your_google_genai_api_key

Install all the required dependencies

pip install langchain langgraph pydantic typing_extensions python-dotenv google-generativeai

Now run the main file python graph.py

