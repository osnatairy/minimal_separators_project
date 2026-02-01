import os
import re
import csv
import lzma
from pathlib import Path

def analyze_gr_file(filepath):
    """
    מנתח קובץ .gr או .gr.xz ומחזיר מידע על הגרף
    """
    try:
        # בדיקה אם הקובץ דחוס
        if filepath.suffix == '.xz' or str(filepath).endswith('.xz'):
            with lzma.open(filepath, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            with open(filepath, 'r') as f:
                lines = f.readlines()

        nodes, edges = 0, 0
        comments = []

        for line in lines:
            line = line.strip()

            # שורות הערות
            if line.startswith('c '):
                comments.append(line[2:].strip())

            # מידע על צמתים
            elif line.startswith('p tw '):
                parts = line.split()
                if len(parts) >= 3:
                    nodes = int(parts[2])
                    edges = int(parts[3])
                break

            # פורמט חלופי - בודק שורות ישירות
            elif 'nodes' in line.lower():
                match = re.search(r'(\d+)', line)
                if match:
                    nodes = int(match.group(1))
            elif 'edges' in line.lower():
                match = re.search(r'(\d+)', line)
                if match:
                    edges = int(match.group(1))

        return {
            'filename': os.path.basename(filepath),
            'nodes': nodes,
            'edges': edges,
            'comments': comments
        }

    except Exception as e:
        return {
            'filename': os.path.basename(filepath),
            'nodes': -1,
            'edges': -1,
            'error': str(e)
        }

def analyze_pace_directory(directory_path):
    """
    מנתח את כל קבצי ה-.gr ו-.gr.xz בתיקייה
    """
    results = []

    # חיפוש קבצי .gr ו-.gr.xz
    for pattern in ['*.gr', '*.gr.xz']:
        for filepath in Path(directory_path).rglob(pattern):
            print(f"מנתח: {filepath.name}")
            result = analyze_gr_file(filepath)
            results.append(result)

    return results

def quick_file_scan(filepath):
    """
    סריקה מהירה רק של השורות הראשונות (תומך ב-xz)
    """
    try:
        if filepath.suffix == '.xz' or str(filepath).endswith('.xz'):
            with lzma.open(filepath, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 20:  # רק 20 שורות ראשונות
                        break

                    line = line.strip()
                    if line.startswith('p tw '):
                        parts = line.split()
                        if len(parts) >= 4:
                            return int(parts[2]), int(parts[3])
        else:
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if i > 20:  # רק 20 שורות ראשונות
                        break

                    line = line.strip()
                    if line.startswith('p tw '):
                        parts = line.split()
                        if len(parts) >= 4:
                            return int(parts[2]), int(parts[3])

        return None, None
    except:
        return None, None

def export_to_csv(results, output_file='named-graphs_sizes.csv'):
    """
    מייצא תוצאות ל-CSV
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'nodes', 'edges', 'density', 'category']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            if result['nodes'] > 0:
                density = result['edges'] / (result['nodes'] * (result['nodes'] - 1) / 2) if result['nodes'] > 1 else 0

                if result['nodes'] < 50:
                    category = 'small'
                elif result['nodes'] < 500:
                    category = 'medium'
                else:
                    category = 'large'

                writer.writerow({
                    'filename': result['filename'],
                    'nodes': result['nodes'],
                    'edges': result['edges'],
                    'density': round(density, 4),
                    'category': category
                })

    print(f"\nנתונים יוצאו ל: {output_file}")
