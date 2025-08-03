import os 
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
from typing import TypedDict , Annotated
from pydantic import BaseModel , Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder , PromptTemplate
import json
from langchain_core.messages import messages_to_dict

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    question: Annotated[str, None]
    intent_classification: Annotated[str, None]
    sales_resolution: Annotated[str, None]
    urgency_classification: Annotated[str, None]
    billing_resolution: Annotated[str, None]
    escalation: Annotated[str, None]
    automated_support: Annotated[str, None]

model = ChatOpenAI(model='gpt-4.1-mini', temperature=0)

class IntentClassification(BaseModel):
    """
        classify the intent of the user's message. 
    """
    intent: Literal["Sales Inquiry", "Technical Support", "Billing Question"] = Field(description="Intent of the user's message")

intent_classification_llm = model.with_structured_output(IntentClassification)

class SalesResolution(BaseModel):
    """
        Generate appropriate response on the user's question related to sales.
    """
    response: str = Field(description="The response to the user's sales related question")

sales_resolution_llm = model.with_structured_output(SalesResolution)

class UrgencyClassificationResolution(BaseModel):
    """
        Find urgency on the user's question related to technical support.
    """
    urgency: Literal["High Urgency", "Standard Urgency"] = Field(description="The urgency for the user's Technical Support question")

urgency_classification_resolution_llm = model.with_structured_output(UrgencyClassificationResolution)

class Escalation(BaseModel):
    """
        Acknowledge issue and let user know that issue will be assigned to customer support agent.
    """
    response: str = Field(description="Acknowledges user's message and assign it to customer support agent")

escalation_llm = model.with_structured_output(Escalation)

class AutomatedSupport(BaseModel):
    """
        Reply to customer with help, FAQs
    """
    response: str = Field(description="Reply to customer with solutions, FAQ, help")

automated_support_llm = model.with_structured_output(AutomatedSupport)

class BillingResolution(BaseModel):
    """
        Generate appropriate response on the user's question related to billing question.
    """
    response: str = Field(description="The response for the user's billing related question")

billing_resolution_llm = model.with_structured_output(BillingResolution)


def init(state):
    """
        Initialize the state of the graph.
    """
    return {
        "messages": [],
        "intent_classification": None,
        "sales_resolution": None,
        "urgency_classification": None,
        "billing_resolution": None,
        "escalation": None,
        "automated_support": None
    }


def intent_classification(state):
    """
       Classify the intent of the user's message. 
    """
    user_input = state["messages"][0]

    prompt = PromptTemplate(
        template="""You are a customer support agent. You are given a user's message.
        You need to classify the intent of the user's message {user_input}
        The intent can be either Sales Inquiry or Technical Support.
        Respond in json format with the following keys:
        intent: The intent of the user's message""",
        input_variables= ["user_input"]
    )

    intent_chain = prompt | intent_classification_llm

    intent_output = intent_chain.invoke({"user_input": user_input})

    return {"intent_classification": intent_output.intent,
            "messages": [AIMessage(content=intent_output.intent)],
            "question": user_input.content}

def sales_resolution(state):
    """
       Generate response for the user's sales related question. 
    """
    user_input = state["messages"][0]

    prompt = PromptTemplate(
        template="""You are a customer support agent. You are given a user's message.
        You need to generate response for the user's message {user_input}
        The response should be in friendly and helpful manner.
        Respond in json format with the following keys:
        response: The response of the user's message""",
        input_variables= ["user_input"]
    )

    sales_chain = prompt | sales_resolution_llm

    sales_chain_output = sales_chain.invoke({"user_input": user_input})

    return {"sales_resolution": sales_chain_output.response,
            "messages": [AIMessage(content=sales_chain_output.response)]}

def urgency_classification(state):
    """
        Find the urgency of  the user's Technical support related question. 
    """
    user_input = state["messages"][0]

    prompt = PromptTemplate(
        template="""You are a customer support agent. You are given a user's message.
        You need to generate response for the user's message {user_input}
        The response should be in friendly and helpful manner.
        Respond in json format with the following keys:
        urgency: The urgency of the user's message""",
        input_variables= ["user_input"]
    )

    urgency_classification_chain = prompt | urgency_classification_resolution_llm

    urgency_output = urgency_classification_chain.invoke({"user_input": user_input})

    return {"urgency_classification": urgency_output.urgency,
            "messages": [AIMessage(content=urgency_output.urgency)]}

def escalation(state):
    """
       Acknowledge customer issue and respond to user that ticket will be created and customer support will contact 
    """
    user_input = state["messages"][0]

    prompt = PromptTemplate(
        template="""You are a customer support agent. You are given a user's message.
        You need to generate response for the user's message {user_input}
        The response should be in friendly and helpful manner.
        Respond in json format with the following keys:
        response: The response of the user's message""",
        input_variables= ["user_input"]
    )

    escalation_chain = prompt | escalation_llm

    escalation_output = escalation_chain.invoke({"user_input": user_input})

    return {"escalation": escalation_output.response,
            "messages": [AIMessage(content=escalation_output.response)]}

def automated_support(state):
    """
       respond to user with help,FAQ
    """
    user_input = state["messages"][0]

    prompt = PromptTemplate(
        template="""You are a customer support agent. You are given a user's message.
        You need to generate response for the user's message {user_input}
        The response should be in friendly and helpful manner.
        Respond in json format with the following keys:
        response: The response of the user's message""",
        input_variables= ["user_input"]
    )

    automated_support_chain = prompt | automated_support_llm

    automated_support_output = automated_support_chain.invoke({"user_input": user_input})

    return {"automated_support": automated_support_output.response,
            "messages": [AIMessage(content=automated_support_output.response)]}

def billing_resolution(state):
    """
       Generate response for the user's billing related question. 
    """
    user_input = state["messages"][0]

    prompt = PromptTemplate(
        template="""You are a customer support agent. You are given a user's message.
        You need to generate response for the user's message {user_input}
        The response should be in friendly and helpful manner.
        Respond in json format with the following keys:
        response: The response of the user's message""",
        input_variables= ["user_input"]
    )

    billing_chain = prompt | billing_resolution_llm

    billing_output = billing_chain.invoke({"user_input": user_input})

    return {"billing_resolution": billing_output.response,
            "messages": [AIMessage(content=billing_output.response)]}

def route_intent(state):
    """
        Route the user's message to appropriate resolution node.
    """
    return state["intent_classification"]

def route_urgency(state):
    """
        Route the user's message to appropriate support channel.
    """
    return state["urgency_classification"]

workflow = StateGraph(GraphState)
workflow.add_node("init", init)
workflow.add_node("intent_classification", intent_classification)
workflow.add_node("sales_resolution", sales_resolution)
workflow.add_node("urgency_classification", urgency_classification)
workflow.add_node("escalation", escalation)
workflow.add_node("automated_support", automated_support)
workflow.add_node("billing_resolution", billing_resolution)

workflow.add_edge(START, "init")
workflow.add_edge("init", "intent_classification")
workflow.add_conditional_edges("intent_classification",
                               route_intent,
                               {
                                   "Sales Inquiry": "sales_resolution",
                                   "Technical Support": "urgency_classification",
                                   "Billing Question": "billing_resolution"
                               })
workflow.add_conditional_edges("urgency_classification",route_urgency,
                               {
                                   "High Urgency": "escalation",
                                   "Standard Urgency": "automated_support"
                               }
                               
                               )
workflow.add_edge("sales_resolution", END)
workflow.add_edge("billing_resolution", END)
app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke(
        {"messages":[HumanMessage(content="can you please let me know how to check credits in chatgpt billing")]}
    )
    serializable_result = result.copy()
    print(serializable_result)
    print("===========================================================================")
    serializable_result['messages'] = messages_to_dict(serializable_result['messages'])
    print(serializable_result)
    print("=============================================================================")
    with open("result.json", "w") as f:
        json.dump(serializable_result, f, indent=2)
    print(result)













