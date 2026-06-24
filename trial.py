from edgar import set_identity, Company

# 1. Authenticate with the SEC
set_identity("Adnan Saifee adnan.saifee2006@gmail.com")

# 2. Get the latest 10-K filing
company = Company("MSFT")
filings = company.get_filings(form="10-K")
filing = filings.latest()
print(filing.period_of_report)

# # 3. Transform the raw filing into a structured Data Object (.obj())
# tenk_object = filing.obj()
# print(tenk_object.__repr__())
# # 4. Extract the exact sections using their standard Item names
# # edgartools handles the boundaries and strips out the structural layout noise for you!
# risk_factors_text = tenk_object["Item 1A"]
# mda_text = tenk_object["Item 7"]

# # 5. Save the cleanly parsed sections to your output files
# with open("risk_factors.txt", "w", encoding="utf-8") as f:
#     f.write(str(risk_factors_text))

# with open("mda.txt", "w", encoding="utf-8") as f:
#     f.write(str(mda_text))

# print("Successfully extracted Item 1A and Item 7 text blocks directly into files!")