from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.tools import tool
from composio_langchain import Action, App, ComposioToolSet
from langchain_openai import ChatOpenAI
import chainlit as cl
from chainlit import run_sync

# Initialize Composio ToolSet
composio_toolset = ComposioToolSet()
tools = composio_toolset.get_tools(apps=[App.RAGTOOL])
hl = HumanLayer(
    run_id="crewai-chatpdf"
)

human_chat = hl.human_as_tool()

# @tool("Ask Human follow up questions")
#     def ask_human(question: str) -> str:
#    		 """Ask human follow up questions"""
#     	human_response  = run_sync( cl.AskUserMessage(content=f"{question}").send())
#    		if human_response:
#        		return human_response["output"]

@CrewBase
class PdfAgent():
    """PdfAgent crew for handling PDF content with RAG."""

    # YAML configuration files for agents and tasks
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
	

    @agent
    def pdf_processor(self) -> Agent:
        """Agent to process and add PDF content to the RAG tool."""
        return Agent(
            config=self.agents_config['pdf_processor'],
            tools=tools,  # Add RAG tool
            verbose=True
        )

    @agent
    def query_agent(self) -> Agent:
        """Agent to query the RAG tool for relevant information."""
        return Agent(
            config=self.agents_config['query_agent'],
            tools=tools,  # Add RAG tool
            verbose=True
        )

    @agent
    def chat_agent(self) -> Agent:
        """Agent to handle conversational interactions based on PDF content."""
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        return Agent(
            role='Chat Assistant',
            goal='Answer user queries based on PDF content and provide insights.',
            backstory="An AI-powered assistant trained on PDF data, helping users find information efficiently.",
            tools=[human_chat], 
            llm=llm,
            verbose=True
        )

    @task
    def process_pdf_task(self) -> Task:
        """Task to extract and add PDF content to the RAG tool."""
        return Task(
            config=self.tasks_config['process_pdf_task'],
            agent=self.pdf_processor(),
        )

    @task
    def query_rag_task(self) -> Task:
        """Task to query the RAG tool for relevant information."""
        return Task(
            config=self.tasks_config['query_rag_task'],
            agent=self.query_agent(),
            output_file='rag_results.md'
        )

    @task
    def chat_task(self) -> Task:
        """Task to handle user queries in a conversational format."""
        return Task(
            description="Engage in chat-based interactions using PDF content.",
            agent=self.chat_agent(),
            expected_output="Relevant answers based on PDF context."
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PdfAgent crew for RAG-based PDF processing and chat."""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
