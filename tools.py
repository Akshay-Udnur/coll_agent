import json
import uuid
import mock_db

def customer_verify(customer_id: str, dob: str, phone: str) -> str:
    """Customer Identification and verification."""
    customer = mock_db.CUSTOMERS.get(customer_id)
    if customer and customer["dob"] == dob and customer["phone"] == phone:
        return json.dumps({"isverified": True, "customer_id": customer_id, "loan_no": customer["loan_no"]})
    return json.dumps({"isverified": False})

def loan_document_lookup(loanno: str, loan_start_date: str) -> str:
    """To search for the exact loan document applicable for the loan."""
    loan = mock_db.LOANS.get(loanno)
    if loan and loan["start_date"] == loan_start_date:
        return json.dumps({"loanDocument": loan["loan_document_id"]})
    return json.dumps({"loanDocument": None})

def loan_document_context_lookup(loanDocument: str) -> str:
    """Lookup the loan document using RAG (simulated) and provide results."""
    for loan in mock_db.LOANS.values():
        if loan["loan_document_id"] == loanDocument:
            return json.dumps({"context": loan["loan_details"]})
    return json.dumps({"context": []})

def fetch_dues(loanno: str) -> str:
    """Checks dues for a loan and fetch amount due, due date and status."""
    loan = mock_db.LOANS.get(loanno)
    if loan:
        return json.dumps({"amount": loan["amount_due"], "due_date": loan["due_date"], "status": loan["status"]})
    return json.dumps({"error": "Loan not found"})

def concession_eligibility(loanno: str, customer_details: str, debt_to_asset_ratio: float, monthly_shortfall: float) -> str:
    """fetch the eligibility rules based on the conversation and the input details."""
    concession = mock_db.CONCESSIONS.get(loanno)
    if concession:
        return json.dumps({"concession": concession["eligibility_options"]})
    return json.dumps({"concession": []})

def concession_plan_fetching(loanno: str, customerID: str) -> str:
    """fetch the different plans based on rules document."""
    concession = mock_db.CONCESSIONS.get(loanno)
    if concession:
        return json.dumps({"concession": concession["plans"]})
    return json.dumps({"concession": []})

def promise_capture(isPTP: bool, due_date: str, amount: float) -> str:
    """Captures promise of customers in json db."""
    if isPTP:
        mock_db.ACTIONS["promises"].append({"due_date": due_date, "amount": amount})
        return json.dumps({"isSuccess": True})
    return json.dumps({"isSuccess": False})

def payment_pause_tool(from_date: str, to_date: str, isVulnerability: bool, isPause: bool) -> str:
    """Captures payment pause if requested by customer."""
    if isPause and isVulnerability:
        mock_db.ACTIONS["payment_pauses"].append({"from": from_date, "to": to_date})
        return json.dumps({"isSuccess": True})
    return json.dumps({"isSuccess": False})

def payment_link_create(isPayNow: bool) -> str:
    """generate payment link with 60 minutes expiration."""
    if isPayNow:
        link = f"https://pay.example.com/{uuid.uuid4().hex[:8]}"
        mock_db.ACTIONS["payment_links"].append(link)
        return json.dumps({"link": link, "expiration": 3600})
    return json.dumps({"error": "Payment not requested"})


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "customer_verify",
            "description": "Customer Identification and verification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Selected customer id from context"},
                    "dob": {"type": "string", "description": "YYYY-MM-DD"},
                    "phone": {"type": "string", "description": "Phone number digits"}
                },
                "required": ["customer_id", "dob", "phone"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loan_document_lookup",
            "description": "To search for the exact loan document that is applicable for the loan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loanno": {"type": "string"},
                    "loan_start_date": {"type": "string", "description": "YYYY-MM-DD"}
                },
                "required": ["loanno", "loan_start_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "loan_document_context_lookup",
            "description": "This will lookup the loan document and provide context results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loanDocument": {"type": "string"}
                },
                "required": ["loanDocument"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_dues",
            "description": "Fetch amount due and due date for a given loan number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loanno": {"type": "string"}
                },
                "required": ["loanno"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "concession_eligibility",
            "description": "fetch the eligibility rules based on the conversation and the input details for hardship/vulnerability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loanno": {"type": "string"},
                    "customer_details": {"type": "string"},
                    "debt_to_asset_ratio": {"type": "number"},
                    "monthly_shortfall": {"type": "number"}
                },
                "required": ["loanno", "customer_details", "debt_to_asset_ratio", "monthly_shortfall"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "concession_plan_fetching",
            "description": "fetch the different plans based on rules document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loanno": {"type": "string"},
                    "customerID": {"type": "string"}
                },
                "required": ["loanno", "customerID"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "promise_capture",
            "description": "Captures promise to pay of customers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "isPTP": {"type": "boolean"},
                    "due_date": {"type": "string", "description": "DD-MM-YYYY"},
                    "amount": {"type": "number"}
                },
                "required": ["isPTP", "due_date", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "payment_pause_tool",
            "description": "Captures payment pause if requested by customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_date": {"type": "string"},
                    "to_date": {"type": "string"},
                    "isVulnerability": {"type": "boolean"},
                    "isPause": {"type": "boolean"}
                },
                "required": ["from_date", "to_date", "isVulnerability", "isPause"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "payment_link_create",
            "description": "generate payment link with 60 minutes expiration.",
            "parameters": {
                "type": "object",
                "properties": {
                    "isPayNow": {"type": "boolean"}
                },
                "required": ["isPayNow"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "customer_verify": customer_verify,
    "loan_document_lookup": loan_document_lookup,
    "loan_document_context_lookup": loan_document_context_lookup,
    "fetch_dues": fetch_dues,
    "concession_eligibility": concession_eligibility,
    "concession_plan_fetching": concession_plan_fetching,
    "promise_capture": promise_capture,
    "payment_pause_tool": payment_pause_tool,
    "payment_link_create": payment_link_create,
}
