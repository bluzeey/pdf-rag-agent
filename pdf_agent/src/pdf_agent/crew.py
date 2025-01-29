from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from composio_langchain import Action, App, ComposioToolSet

# Initialize Composio ToolSet
composio_toolset = ComposioToolSet()
tools = composio_toolset.get_tools(apps=[App.RAGTOOL])

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

    @crew
    def crew(self) -> Crew:
        """Creates the PdfAgent crew for RAG-based PDF processing."""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )