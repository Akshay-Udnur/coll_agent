import json
from datetime import datetime

# Sample mocked database for our collection agent
CUSTOMERS = {
    "C101": {
        "dob": "1990-01-01",
        "phone": "9876543210",
        "name": "Jane Doe",
        "loan_no": "LN-1001"
    },
    "C102": {
        "dob": "1988-11-15",
        "phone": "9123456780",
        "name": "Ravi Kumar",
        "loan_no": "LN-1002"
    },
    "C103": {
        "dob": "1995-07-21",
        "phone": "9988776655",
        "name": "Aisha Khan",
        "loan_no": "LN-1003"
    }
}

LOANS = {
    "LN-1001": {
        "start_date": "2020-01-01",
        "loan_document_id": "loan_rules_standard_v1.pdf",
        "amount_due": 400.0,
        "due_date": "05-05-2026",
        "status": "Not paid",
        "loan_details": [
            "Standard personal loan terms for late payment and hardship handling.",
            "Eligible for payment pause only with vulnerability evidence."
        ]
    },
    "LN-1002": {
        "start_date": "2021-08-10",
        "loan_document_id": "loan_rules_flexi_v2.pdf",
        "amount_due": 1250.0,
        "due_date": "20-04-2026",
        "status": "Not paid",
        "loan_details": [
            "Flexi loan terms with one no-penalty partial-payment option.",
            "Can request short-term restructuring when monthly shortfall is proven."
        ]
    },
    "LN-1003": {
        "start_date": "2023-02-18",
        "loan_document_id": "loan_rules_prime_v1.pdf",
        "amount_due": 780.0,
        "due_date": "28-04-2026",
        "status": "Partially paid",
        "loan_details": [
            "Prime loan terms with stricter delinquency follow-up.",
            "Concessions available only for verified hardship."
        ]
    }
}

CONCESSIONS = {
    "LN-1001": {
        "eligibility_options": ["pause_3_months", "restructure_6_months"],
        "plans": [
            {"plan_id": "PausePlan1", "description": "Pause payments for 3 months."},
            {"plan_id": "RestructurePlan1", "description": "Restructure loan by extending term by 6 months."}
        ]
    },
    "LN-1002": {
        "eligibility_options": ["partial_pay_2_cycles", "restructure_3_months"],
        "plans": [
            {"plan_id": "FlexiPartial2", "description": "Allow partial payments for next 2 billing cycles."},
            {"plan_id": "FlexiRestructure3", "description": "Restructure loan by extending term by 3 months."}
        ]
    },
    "LN-1003": {
        "eligibility_options": ["hardship_review_only", "restructure_1_month"],
        "plans": [
            {"plan_id": "PrimeReview", "description": "Route to hardship review before concession approval."},
            {"plan_id": "PrimeRestructure1", "description": "One-month term extension after hardship confirmation."}
        ]
    }
}

# State DB to capture actions
ACTIONS = {
    "promises": [],
    "payment_pauses": [],
    "payment_links": []
}
