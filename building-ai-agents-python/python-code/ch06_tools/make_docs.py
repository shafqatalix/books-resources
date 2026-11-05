from lib.config import DOCS_DIR

DOCS = {
    "refund_policy.txt": (
        "Acme Cloud refund policy. Customers may request a full refund within "
        "30 days of purchase. After 30 days, annual plans are refunded for the "
        "unused full months. Monthly plans are not refundable after the first "
        "30 days. Send refund requests to billing@acme.example. Requests are "
        "answered within 3 business days."
    ),
    "onboarding.txt": (
        "Welcome to Acme Cloud. On your first day you will receive a laptop and "
        "an account setup email. Complete the security training in your first "
        "week. Your manager will assign a buddy for your first month. The team "
        "standup is at 9:30 each weekday. Request time off through the HR "
        "portal at least two weeks ahead."
    ),
    "security.txt": (
        "Acme Cloud security guidelines. Use a password manager and enable "
        "two-factor authentication on every work account. Never share "
        "credentials in chat. Report suspected phishing to security@acme.example "
        "within one hour. Laptops must use full-disk encryption. Customer data "
        "may only be accessed through approved admin tools."
    ),
}

DOCS_DIR.mkdir(exist_ok=True)
for name, text in DOCS.items():
    (DOCS_DIR / name).write_text(text, encoding="utf-8")
print("Wrote", len(DOCS), "documents to", DOCS_DIR)
