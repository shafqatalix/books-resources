from lib.tools import list_documents, read_document

print(list_documents.invoke({}))
print(read_document.invoke({"filename": "refund_policy.txt"}))
print(read_document.invoke({"filename": "../.env"}))
