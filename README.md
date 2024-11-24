# COMPSCI646PROJCT

Instructions on getting baseline results
1. Download HealthVer dataset https://github.com/sarrouti/HealthVer/tree/master and run getClaims.py to generate claims.csv
1. Download 2020-07-16 version data from CORD 19 dataset https://github.com/allenai/cord19/tree/master?tab=readme-ov-file
2. Download qrels file from https://ir.nist.gov/covidSubmit/data.html file
3. Run process_metadata.py, then process_qrels.py and then run evaluate_baseline.py to get results. 