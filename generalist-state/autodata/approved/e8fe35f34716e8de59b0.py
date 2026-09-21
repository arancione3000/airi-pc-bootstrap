import os
import sys

# Add parent directory to path so we can import the local sec_api package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load API key from .env file in project root
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
api_key = ""
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.startswith("SEC_API_IO_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

if not api_key:
    api_key = os.environ.get("SEC_API_IO_API_KEY", "")

if not api_key:
    print("No API key found. Set SEC_API_IO_API_KEY in .env or environment.")
    sys.exit(1)

from sec_api import (
    QueryApi,
    FullTextSearchApi,
    RenderApi,
    DownloadApi,
    XbrlApi,
    ExtractorApi,
    MappingApi,
    InsiderTradingApi,
    Form144Api,
    Form13FHoldingsApi,
    Form13FCoverPagesApi,
    Form13DGApi,
    FormNportApi,
    FormNcenApi,
    FormNPXApi,
    Form_S1_424B4_Api,
    FormDApi,
    FormCApi,
    RegASearchAllApi,
    Form_8K_Item_X_Api,
    FormAdvApi,
    ExecCompApi,
    DirectorsBoardMembersApi,
    FloatApi,
    SubsidiaryApi,
    SecEnforcementActionsApi,
    SecLitigationsApi,
    SecAdministrativeProceedingsApi,
    AaerApi,
    SroFilingsApi,
    EdgarEntitiesApi,
    AuditFeesApi,
    EdgarIndexApi,
    Datasets,
)

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  \u2705 {name}")
    except Exception as err:
        failed += 1
        print(f"  \u274c {name}")
        print(f"    {err}")


# ── Query API ──────────────────────────────────────────────

print("\nQuery API")
queryApi = QueryApi(api_key=api_key)

def test_query_api():
    result = queryApi.get_filings({
        "query": 'formType:"10-Q" AND ticker:AAPL',
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert result["filings"] and len(result["filings"]) > 0, "No filings returned"

test("get_filings returns results", test_query_api)


# ── Full-Text Search API ──────────────────────────────────

print("\nFull-Text Search API")
fullTextSearchApi = FullTextSearchApi(api_key=api_key)

def test_full_text_search():
    result = fullTextSearchApi.get_filings({
        "query": '"LPCN 1154"',
        "formTypes": ["8-K", "10-Q"],
        "startDate": "2021-01-01",
        "endDate": "2021-06-14",
    })
    assert result["filings"] and len(result["filings"]) > 0, "No filings returned"

test("get_filings returns results", test_full_text_search)


# ── Download API ──────────────────────────────────────────

print("\nDownload API")
downloadApi = DownloadApi(api_key=api_key)

def test_download_api():
    content = downloadApi.get_file(
        "https://www.sec.gov/Archives/edgar/data/1318605/000162828025045968/tsla-20250930.htm"
    )
    assert isinstance(content, str), "Expected string content"
    assert len(content) > 1000, "Content too short"

test("get_file downloads filing content", test_download_api)


# ── XBRL-to-JSON API ─────────────────────────────────────

print("\nXBRL-to-JSON API")
xbrlApi = XbrlApi(api_key=api_key)

def test_xbrl_api():
    result = xbrlApi.xbrl_to_json(accession_no="0000320193-20-000096")
    assert "CoverPage" in result, "Missing CoverPage"
    assert "StatementsOfIncome" in result, "Missing StatementsOfIncome"

test("xbrl_to_json converts by accession number", test_xbrl_api)


# ── Extractor API ─────────────────────────────────────────

print("\nExtractor API")
extractorApi = ExtractorApi(api_key=api_key)

def test_extractor_api():
    text = extractorApi.get_section(
        "https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-10k_20201231.htm",
        "1A",
        "text",
    )
    assert isinstance(text, str), "Expected string"
    assert len(text) > 100, "Section text too short"

test("get_section extracts 10-K section", test_extractor_api)


# ── Mapping API ───────────────────────────────────────────

print("\nMapping API")
mappingApi = MappingApi(api_key=api_key)

def test_mapping_api():
    result = mappingApi.resolve("ticker", "TSLA")
    assert isinstance(result, list), "Expected list"
    assert len(result) > 0, "No results"
    assert result[0]["ticker"] == "TSLA", "Wrong ticker"

test("resolve ticker returns company data", test_mapping_api)


# ── Insider Trading API ───────────────────────────────────

print("\nInsider Trading API")
insiderTradingApi = InsiderTradingApi(api_key=api_key)

def test_insider_trading():
    result = insiderTradingApi.get_data({
        "query": "issuer.tradingSymbol:TSLA",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert result["transactions"] and len(result["transactions"]) > 0, "No transactions returned"

test("get_data returns insider transactions", test_insider_trading)


# ── Form 144 API ──────────────────────────────────────────

print("\nForm 144 API")
form144Api = Form144Api(api_key=api_key)

def test_form144():
    result = form144Api.get_data({
        "query": "entities.ticker:TSLA",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns Form 144 filings", test_form144)


# ── Form 13F Holdings API ────────────────────────────────

print("\nForm 13F Holdings API")
form13FHoldingsApi = Form13FHoldingsApi(api_key=api_key)

def test_form13f_holdings():
    result = form13FHoldingsApi.get_data({
        "query": "cik:1067983",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns 13F holdings", test_form13f_holdings)


# ── Form 13F Cover Pages API ─────────────────────────────

print("\nForm 13F Cover Pages API")
form13FCoverPagesApi = Form13FCoverPagesApi(api_key=api_key)

def test_form13f_cover_pages():
    result = form13FCoverPagesApi.get_data({
        "query": "cik:1067983",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns 13F cover pages", test_form13f_cover_pages)


# ── Form 13D/13G API ─────────────────────────────────────

print("\nForm 13D/13G API")
form13DGApi = Form13DGApi(api_key=api_key)

def test_form13dg():
    result = form13DGApi.get_data({
        "query": "accessionNo:*",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "filings" in result, "No filings property"

test("get_data returns 13D/13G filings", test_form13dg)


# ── Form N-PORT API ───────────────────────────────────────

print("\nForm N-PORT API")
nportApi = FormNportApi(api_key=api_key)

def test_nport():
    result = nportApi.get_data({
        "query": "fundInfo.totAssets:[100000000 TO *]",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert result["filings"] and len(result["filings"]) > 0, "No filings returned"

test("get_data returns N-PORT filings", test_nport)


# ── Form N-CEN API ────────────────────────────────────────

print("\nForm N-CEN API")
formNcenApi = FormNcenApi(api_key=api_key)

def test_ncen():
    result = formNcenApi.get_data({
        "query": "accessionNo:*",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns N-CEN filings", test_ncen)


# ── Form N-PX API ─────────────────────────────────────────

print("\nForm N-PX API")
formNpxApi = FormNPXApi(api_key=api_key)

def test_npx_metadata():
    result = formNpxApi.get_metadata({
        "query": "cik:884546",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_metadata returns N-PX metadata", test_npx_metadata)


# ── Form S-1/424B4 API ───────────────────────────────────

print("\nForm S-1/424B4 API")
formS1Api = Form_S1_424B4_Api(api_key=api_key)

def test_form_s1():
    result = formS1Api.get_data({
        "query": "ticker:RIVN",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns S-1 filings", test_form_s1)


# ── Form D API ────────────────────────────────────────────

print("\nForm D API")
formDApi = FormDApi(api_key=api_key)

def test_form_d():
    result = formDApi.get_data({
        "query": "offeringData.offeringSalesAmounts.totalOfferingAmount:[1000000 TO *]",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "offerings" in result, "No offerings property"

test("get_data returns Form D filings", test_form_d)


# ── Form C API ────────────────────────────────────────────

print("\nForm C API")
formCApi = FormCApi(api_key=api_key)

def test_form_c():
    result = formCApi.get_data({
        "query": "id:*",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns Form C filings", test_form_c)


# ── Reg A Search API ──────────────────────────────────────

print("\nReg A Search API")
regASearchAllApi = RegASearchAllApi(api_key=api_key)

def test_reg_a():
    result = regASearchAllApi.get_data({
        "query": "filedAt:[2024-01-01 TO 2024-12-31]",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns Reg A filings", test_reg_a)


# ── Form 8-K API ──────────────────────────────────────────

print("\nForm 8-K API")
form8KApi = Form_8K_Item_X_Api(api_key=api_key)

def test_form_8k():
    result = form8KApi.get_data({
        "query": "item4_01:* AND filedAt:[2024-01-01 TO 2024-12-31]",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns 8-K filings", test_form_8k)


# ── Form ADV API ──────────────────────────────────────────

print("\nForm ADV API")
formAdvApi = FormAdvApi(api_key=api_key)

def test_adv_firms():
    result = formAdvApi.get_firms({
        "query": 'Info.BusNm:"Bridgewater"',
        "from": "0",
        "size": "1",
        "sort": [{"Info.FirmCrdNb": {"order": "desc"}}],
    })
    assert result["filings"] and len(result["filings"]) > 0, "No filings returned"

test("get_firms returns advisory firms", test_adv_firms)

def test_adv_individuals():
    result = formAdvApi.get_individuals({
        "query": "CrntEmps.CrntEmp.orgPK:149777",
        "from": "0",
        "size": "1",
        "sort": [{"id": {"order": "desc"}}],
    })
    assert result["filings"] and len(result["filings"]) > 0, "No filings returned"

test("get_individuals returns individual advisors", test_adv_individuals)

def test_adv_direct_owners():
    result = formAdvApi.get_direct_owners(crd="361")
    assert isinstance(result, list) and len(result) > 0, "No direct owners returned"

test("get_direct_owners returns Schedule A data", test_adv_direct_owners)

def test_adv_indirect_owners():
    result = formAdvApi.get_indirect_owners(crd="149777")
    assert isinstance(result, list) and len(result) > 0, "No indirect owners returned"

test("get_indirect_owners returns Schedule B data", test_adv_indirect_owners)

def test_adv_private_funds():
    result = formAdvApi.get_private_funds(crd="793")
    assert isinstance(result, list) and len(result) > 0, "No private funds returned"

test("get_private_funds returns Schedule D 7.B.1 data", test_adv_private_funds)

def test_adv_brochures():
    result = formAdvApi.get_brochures(149777)
    assert result["brochures"] and len(result["brochures"]) > 0, "No brochures returned"

test("get_brochures returns brochure data", test_adv_brochures)

def test_adv_other_business_names():
    result = formAdvApi.get_other_business_names(crd="149777")
    assert isinstance(result, list) and len(result) > 0, "No other business names returned"

test("get_other_business_names returns Schedule D 1.B data", test_adv_other_business_names)

def test_adv_separately_managed_accounts():
    result = formAdvApi.get_separately_managed_accounts(crd="149777")
    assert result, "No separately managed accounts data returned"

test("get_separately_managed_accounts returns Schedule D 5.K data", test_adv_separately_managed_accounts)

def test_adv_financial_industry_affiliations():
    result = formAdvApi.get_financial_industry_affiliations(crd="149777")
    assert isinstance(result, list) and len(result) > 0, "No financial industry affiliations returned"

test("get_financial_industry_affiliations returns Schedule D 7.A data", test_adv_financial_industry_affiliations)


# ── Executive Compensation API ────────────────────────────

print("\nExecutive Compensation API")
execCompApi = ExecCompApi(api_key=api_key)

def test_exec_comp():
    result = execCompApi.get_data("TSLA")
    assert isinstance(result, list), "Expected list"
    assert len(result) > 0, "No results"

test("get_data by ticker returns compensation data", test_exec_comp)


# ── Directors & Board Members API ─────────────────────────

print("\nDirectors & Board Members API")
directorsBoardMembersApi = DirectorsBoardMembersApi(api_key=api_key)

def test_directors():
    result = directorsBoardMembersApi.get_data({
        "query": "ticker:AAPL",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns directors data", test_directors)


# ── Float API ─────────────────────────────────────────────

print("\nFloat API")
floatApi = FloatApi(api_key=api_key)

def test_float():
    result = floatApi.get_float(ticker="AAPL")
    assert result and "data" in result, "No data returned"

test("get_float returns share data", test_float)


# ── Subsidiary API ────────────────────────────────────────

print("\nSubsidiary API")
subsidiaryApi = SubsidiaryApi(api_key=api_key)

def test_subsidiary():
    result = subsidiaryApi.get_data({
        "query": "ticker:AAPL",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns subsidiary data", test_subsidiary)


# ── SEC Enforcement Actions API ───────────────────────────

print("\nSEC Enforcement Actions API")
enforcementActionsApi = SecEnforcementActionsApi(api_key=api_key)

def test_enforcement():
    result = enforcementActionsApi.get_data({
        "query": "releasedAt:[2024-01-01 TO 2024-12-31]",
        "from": "0",
        "size": "1",
        "sort": [{"releasedAt": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns enforcement actions", test_enforcement)


# ── SEC Litigation Releases API ───────────────────────────

print("\nSEC Litigation Releases API")
secLitigationsApi = SecLitigationsApi(api_key=api_key)

def test_litigations():
    result = secLitigationsApi.get_data({
        "query": "releasedAt:[2024-01-01 TO 2024-12-31]",
        "from": "0",
        "size": "1",
        "sort": [{"releasedAt": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns litigation releases", test_litigations)


# ── SEC Administrative Proceedings API ────────────────────

print("\nSEC Administrative Proceedings API")
adminProceedingsApi = SecAdministrativeProceedingsApi(api_key=api_key)

def test_admin_proceedings():
    result = adminProceedingsApi.get_data({
        "query": "releasedAt:[2024-01-01 TO 2024-12-31]",
        "from": "0",
        "size": "1",
        "sort": [{"releasedAt": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns admin proceedings", test_admin_proceedings)


# ── AAER API ──────────────────────────────────────────────

print("\nAAER API")
aaerApi = AaerApi(api_key=api_key)

def test_aaer():
    result = aaerApi.get_data({
        "query": "dateTime:[2020-01-01 TO 2024-12-31]",
        "from": "0",
        "size": "1",
        "sort": [{"dateTime": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns AAERs", test_aaer)


# ── SRO Filings API ──────────────────────────────────────

print("\nSRO Filings API")
sroFilingsApi = SroFilingsApi(api_key=api_key)

def test_sro():
    result = sroFilingsApi.get_data({
        "query": "sro:NYSE",
        "from": "0",
        "size": "1",
        "sort": [{"issueDate": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns SRO filings", test_sro)


# ── EDGAR Entities API ────────────────────────────────────

print("\nEDGAR Entities API")
edgarEntitiesApi = EdgarEntitiesApi(api_key=api_key)

def test_edgar_entities():
    result = edgarEntitiesApi.get_data({
        "query": 'name:"Tesla"',
        "from": "0",
        "size": "1",
        "sort": [{"cikUpdatedAt": {"order": "desc"}}],
    })
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_data returns entities", test_edgar_entities)


# ── Audit Fees API ────────────────────────────────────────

print("\nAudit Fees API")
auditFeesApi = AuditFeesApi(api_key=api_key)

def test_audit_fees():
    result = auditFeesApi.get_data({
        "query": "cik:1318605",
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    })
    assert "data" in result, "No data property"

test("get_data returns audit fees", test_audit_fees)


# ── EDGAR Index Ingestion Log API ─────────────────────────

print("\nEDGAR Index Ingestion Log API")
edgarIndexApi = EdgarIndexApi(api_key=api_key)

def test_edgar_index():
    result = edgarIndexApi.get_ingestion_log("2025-12-02")
    assert result["data"] and len(result["data"]) > 0, "No data returned"

test("get_ingestion_log returns filings for a date", test_edgar_index)


# ── Datasets API ──────────────────────────────────────────

print("\nDatasets API")
datasets = Datasets(api_key=api_key)

def test_datasets_get_all():
    result = datasets.get_all()
    assert isinstance(result, list), "Expected list"
    assert len(result) > 0, "No datasets returned"
    assert "datasetIdInUrl" in result[0], "Missing datasetIdInUrl"

test("get_all returns available datasets", test_datasets_get_all)

def test_datasets_get_details():
    result = datasets.get_dataset_details("audit-fees")
    assert "datasetId" in result, "Missing datasetId"
    assert "containers" in result, "Missing containers"
    assert len(result["containers"]) > 0, "No containers"

test("get_dataset_details returns dataset details", test_datasets_get_details)


# ── Summary ───────────────────────────────────────────────

print(f"\n{passed + failed} tests, {passed} passed, {failed} failed\n")
sys.exit(1 if failed > 0 else 0)
