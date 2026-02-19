import requests, textwrap, json

host_url = "http://10.138.0.3:8000" # ampere 9
# host_url = "http://172.24.75.232:8000" # ampere 7
# host_url = "https://app.biomni.stanford.edu/api"

resp = requests.post(f"{host_url}/start_runtime")
print(resp)
session_id = resp.json()["session_id"]

cookie = resp.cookies

print(session_id)

code = textwrap.dedent(r"""
# from biomni.tool.database import query_ensembl
# import boto3
# import pandas as pd
# import s3fs

# # Step 1: Get gene information from Ensembl
# print("Converting Ensembl ID to gene symbol")
# ensembl_result = query_ensembl("Get gene symbol and basic information for ENSG00000080709")
# print("Ensembl query result:")
# print(ensembl_result)

print("=== Querying Databases for DLBCL-Associated Genes ===")
from biomni.tool.database import query_opentarget, query_kegg

# Query OpenTargets for DLBCL associated genes
dlbcl_genes = query_opentarget(
    prompt="Find genes associated with diffuse large B-cell lymphoma DLBCL including oncogenes tumor suppressors and drug targets"
)
print("OpenTargets DLBCL genes result type:", type(dlbcl_genes))
print("Keys in result:", list(dlbcl_genes.keys()) if isinstance(dlbcl_genes, dict) else "Not a dictionary")
print("Result preview:")
print(dlbcl_genes)
print("\n" + "="*30 + "\n")

# from biomni.tool.literature import advanced_web_search
# from biomni.tool.database import query_uniprot, query_kegg, query_opentarget

# # Research selinexor mechanism of action and molecular targets
# print("=== Researching Selinexor Mechanism of Action ===")
# selinexor_search = advanced_web_search(
#     query="selinexor mechanism of action molecular targets nuclear export inhibitor XPO1 DLBCL, include citations",
#     max_searches=3
# )
# print("Selinexor mechanism search result type:", type(selinexor_search))
# print(selinexor_search)
# print("\n" + "="*50 + "\n")


import gget

# print("=== Accessing Biological Data Lake for Cancer Genes ===")
# import boto3
# import pandas as pd
# import s3fs

# # # Initialize S3 filesystem
# fs = s3fs.S3FileSystem()

# # List all files under the S3 bucket 'biomni-datalake'
# print("Listing all files under s3://biomni-datalake ...")
# fs = s3fs.S3FileSystem()
# all_files = fs.ls('biomni-datalake', detail=True)
# for f in all_files:
#     print(f["Key"] if isinstance(f, dict) and "Key" in f else f)
# print(f"Total files found: {len(all_files)}")

print("=" * 80)
print("== Accessing Biological Data Lake for Cancer Genes ==")
import pandas as pd
df = pd.read_parquet("/mnt/biomni_filestore/biomni/biomni_data/data_lake/co-fractionation.parquet")
print(df.head())


import numpy as np
import importlib.metadata as md

print("numpy __version__:", np.__version__)
print("numpy __file__   :", np.__file__)
print("metadata version :", md.version("numpy"))
dist = md.distribution("numpy")
print("metadata location:", dist.locate_file(""))

import importlib.metadata as md

for pkg in ["langchain-openai", "langchain-core"]:
    try:
        version = md.version(pkg)
        dist = md.distribution(pkg)
        location = dist.locate_file("") if dist else "N/A"
        print(f"{pkg} version: {version}")
        print(f"{pkg} location: {location}")
    except Exception as e:
        print(f"{pkg} not found or error: {str(e)}")


# from biomni.tool.literature import query_pubmed

# fdP_papers = query_pubmed("yfdP gene Escherichia coli sequence", max_papers=3)
# print("yfdP literature search results:")
# print(fdP_papers)

# print("=" * 80)

# from biomni.tool.database import query_uniprot, query_kegg
# from biomni.tool.genomics import get_rna_seq_archs4

# # List of candidate genes
# candidate_genes = ["NHLRC2", "MRPL43", "PSMB3", "HRASLS5", "CALB2", 
#                    "RSPRY1", "GUCY2C", "KPNA2", "PYGM", "SMARCC1", "CCDC174"]

# print("=" * 80)
# print("Investigating gene expression in immune-related tissues")
# print("=" * 80)

# # Check expression patterns for each gene
# expression_data = {}
# for gene in candidate_genes:
#     try:
#         result = get_rna_seq_archs4(gene, K=15)
#         expression_data[gene] = result
#         print(f"\n{gene}:")
#         print(result)
#     except Exception as e:
#         print(f"\n{gene}: Error - {str(e)}")


from biomni.tool.database import query_opentarget_genetics
 
genes = ["GABRA3", "GABRE", "GABRQ", "NSDHL", "CETN2", 
        "CSAG1", "CSAG2", "CSAG3", "MAGEA10", "MAGEA2"]
 
print("Querying OpenTargets Genetics for epilepsy associations...\n")
result = query_opentarget_genetics(
    prompt="Find associations between the following genes and epilepsy: GABRA3, GABRE, GABRQ, NSDHL, CETN2, CSAG1, CSAG2, CSAG3",
    verbose=True
)


# from biomni.tool.literature import advanced_web_search

# print("----- Searching specifically for SLCO1B1 variants and bilirubin associations -----\n")

# query = '''SLCO1B1 gene variants rs9637599, rs3780181, rs11119805, and rs4947534:
# - Which is most strongly associated with bilirubin levels in gwas studies?
# - Report p-values and effect sizes (beta coefficients)
# - Include specific studies (e.g., UK Biobank, FinnGen)
# - What is the biological mechanism? (SLCO1B1 encodes OATP1B1 transporter)
# '''

# result = advanced_web_search(query, max_searches=2)
# print(result)
""")

resp = requests.post(
    f"{host_url}/execute",
    json={"code": code, "session_id": session_id},
    cookies=cookie
)
# print(resp.json())
print(resp.json()["output"])

requests.post(f"{host_url}/delete_runtime", json={"session_id": session_id}, cookies=cookie)

