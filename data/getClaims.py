import csv

def process_csv(inputFile, outputFile):
    claimsSet = set()
    with open(inputFile, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            claim = row['claim']
            topicIP = int(row['topic_ip'])
            if topicIP > 50 or '"' in claim:
                continue
            claimsSet.add((claim, topicIP))
    with open(outputFile, mode='w', encoding='utf-8', newline='') as outFile:
        writer = csv.writer(outFile)
        writer.writerow(['claim', 'topic_ip'])
        for claim, topicIP in claimsSet:
            writer.writerow([claim, topicIP])
inputFile =  '/path/to/HealthVer_dev.csv'
outputFile = './data/claims.csv'
process_csv(inputFile, outputFile)
print(f"Filtered CSV file has been saved to: {outputFile}")