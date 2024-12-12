from flask import Flask, request, render_template
import pandas as pd
import os
import pickle
import logging
from collections import defaultdict
from itertools import combinations

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Nama file Pickle
PICKLE_FILE = 'dhp_model.pkl' 

# Fungsi untuk memuat data dari file Pickle
def load_data_from_pickle(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as file:
            return pickle.load(file)
    return None, None

# Fungsi untuk menyimpan data ke file Pickle
def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

# Fungsi DHP yang dioptimalkan
def dhp_optimized(transactions, min_support, min_confidence):
    transactions = [set(transaction) for transaction in transactions if transaction]
    
    logging.debug("Total Transactions after cleaning: %d", len(transactions))
    
    # Single item support count
    support_count = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            support_count[frozenset([item])] += 1
    
    logging.debug("Initial Support Count: %s", support_count)
    
    # Generate frequent 1-itemsets
    frequent_itemsets = []
    current_frequent_itemsets = {itemset: count for itemset, count in support_count.items() if count >= min_support}
    frequent_itemsets.append(current_frequent_itemsets)
    
    logging.debug("Frequent 1-itemsets: %s", current_frequent_itemsets)
    
    k = 1
    while current_frequent_itemsets:
        k += 1
        candidate_itemsets = generate_candidates(current_frequent_itemsets.keys(), k)
        candidate_support_count = defaultdict(int)
        
        for transaction in transactions:
            for candidate in candidate_itemsets:
                if candidate.issubset(transaction):
                    candidate_support_count[candidate] += 1
        
        current_frequent_itemsets = {itemset: count for itemset, count in candidate_support_count.items() if count >= min_support}
        
        if current_frequent_itemsets:
            frequent_itemsets.append(current_frequent_itemsets)
            logging.debug("Frequent %d-itemsets: %s", k, current_frequent_itemsets)
        else:
            logging.debug("No more frequent itemsets found for k=%d", k)
    
    rules = generate_rules(frequent_itemsets, min_confidence, len(transactions))
    
    return frequent_itemsets, rules

def generate_candidates(frequent_itemsets, k):
    items = set()
    for itemset in frequent_itemsets:
        for item in itemset:
            items.add(item)
    return set([frozenset(combination) for combination in combinations(items, k)])

def generate_rules(frequent_itemsets, min_confidence, num_transactions):
    rules = []
    for k, itemset_support in enumerate(frequent_itemsets):
        for itemset, support in itemset_support.items():
            if len(itemset) < 2:
                continue
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    antecedent_support = frequent_itemsets[len(antecedent)-1][antecedent]
                    confidence = support / antecedent_support
                    lift = confidence / (frequent_itemsets[len(consequent)-1][consequent] / num_transactions)
                    if confidence >= min_confidence:
                        rules.append((
                            list(antecedent),  # Convert antecedent to list
                            list(consequent),  # Convert consequent to list
                            support / num_transactions,
                            confidence,
                            lift
                        ))
    return rules

# Fungsi untuk membaca file CSV besar dengan cara yang efisien
def read_large_csv(file):
    chunks = []
    for chunk in pd.read_csv(file, header=None, chunksize=10000):
        chunk = chunk.dropna(how='all')
        chunk_list = chunk.values.tolist()
        cleaned_chunk = [[item for item in transaction if pd.notna(item)] for transaction in chunk_list]
        chunks.extend(cleaned_chunk)
    return chunks

# Muat data dari file Pickle saat aplikasi dijalankan
saved_frequent_itemsets, saved_rules = load_data_from_pickle(PICKLE_FILE)
saved_params = None

if saved_frequent_itemsets:
    saved_params = {'min_support': saved_frequent_itemsets[0].get('min_support'), 'min_confidence': saved_frequent_itemsets[0].get('min_confidence')}

# Tambahkan filter custom untuk format float
@app.template_filter('floatformat')
def floatformat_filter(value, precision=2):
    try:
        return f"{float(value):.{precision}f}"
    except (ValueError, TypeError):
        return value

@app.route('/', methods=['GET', 'POST'])
def index():
    global saved_frequent_itemsets, saved_rules, saved_params
    frequent_itemsets = saved_frequent_itemsets
    rules = saved_rules
    data_uploaded = False
    if request.method == 'POST':
        file = request.files['file']
        start_date = pd.to_datetime(request.form['start_date'])
        end_date = pd.to_datetime(request.form['end_date'])
        min_support_percent = float(request.form['min_support'])  # Input dalam bentuk persen
        min_confidence_percent = float(request.form['min_confidence'])  # Input dalam bentuk persen
        
        min_support = min_support_percent / 100  # Konversi ke bentuk desimal
        min_confidence = min_confidence_percent / 100  # Konversi ke bentuk desimal
        
        logging.debug("File received: %s", file.filename)
        logging.debug("Start Date: %s, End Date: %s", start_date, end_date)
        logging.debug("Min Support: %.2f (%.2f%%), Min Confidence: %.2f (%.2f%%)", min_support, min_support_percent, min_confidence, min_confidence_percent)
        
        if file:
            try:
                data = read_large_csv(file)
                
                # Filter data berdasarkan tanggal
                filtered_data = []
                for row in data:
                    transaction_date = row[0]  # Asumsi tanggal ada di kolom pertama
                    try:
                        transaction_date = pd.to_datetime(transaction_date)
                        if start_date <= transaction_date <= end_date:
                            filtered_data.append(row[1:])  # Sisanya adalah item transaksi
                    except Exception as e:
                        logging.error("Error parsing date: %s", e)
                
                logging.debug("Filtered Data Length: %d", len(filtered_data))
                
                min_support_count = min_support * len(filtered_data)  # Tetap hitung dengan mendasarkan pada jumlah transaksi
                frequent_itemsets, rules = dhp_optimized(filtered_data, min_support=min_support_count, min_confidence=min_confidence)
                logging.debug("Frequent Itemsets: %s", frequent_itemsets)
                logging.debug("Association Rules: %s", rules)
                save_to_pickle((frequent_itemsets, rules), PICKLE_FILE)
                saved_frequent_itemsets = frequent_itemsets
                saved_rules = rules
                saved_params = {'min_support': min_support_percent, 'min_confidence': min_confidence_percent}
                data_uploaded = True
            except Exception as e:
                logging.error("Error processing file: %s", e)
    
    logging.debug("Data Uploaded: %s", data_uploaded)
    return render_template('index.html', frequent_itemsets=frequent_itemsets, rules=rules, data_uploaded=data_uploaded, enumerate=enumerate, list=list)

if __name__ == '__main__':
    app.run(debug=True)