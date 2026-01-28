from pathlib import Path
import os
import subprocess

from minimal_separators_project.io.pace2017_gr import (
    analyze_pace_directory,
    export_to_csv,
    quick_file_scan
)


def create_summary_report(results):
    """
    יוצר דו"ח מסכם
    """
    print("\n" + "=" * 80)
    print("דו\"ח גדלי גרפי PACE 2017")
    print("=" * 80)

    # מיון לפי גודל
    valid_results = [r for r in results if r['nodes'] > 0]
    valid_results.sort(key=lambda x: x['nodes'])

    print(f"\nסה\"כ קבצים נמצאו: {len(results)}")
    print(f"קבצים תקינים: {len(valid_results)}")

    if valid_results:
        print(f"\nסטטיסטיקות:")
        print(f"גרף הקטן ביותר: {valid_results[0]['nodes']} צמתים, {valid_results[0]['edges']} קשתות")
        print(f"גרף הגדול ביותר: {valid_results[-1]['nodes']} צמתים, {valid_results[-1]['edges']} קשתות")

        # התפלגות לפי גדלים
        small_graphs = [r for r in valid_results if r['nodes'] < 50]
        medium_graphs = [r for r in valid_results if 50 <= r['nodes'] < 500]
        large_graphs = [r for r in valid_results if r['nodes'] >= 500]

        print(f"\nהתפלגות גדלים:")
        print(f"גרפים קטנים (< 50 צמתים): {len(small_graphs)}")
        print(f"גרפים בינוניים (50-499 צמתים): {len(medium_graphs)}")
        print(f"גרפים גדולים (≥ 500 צמתים): {len(large_graphs)}")

    print("\n" + "-" * 80)
    print("פירוט קבצים:")
    print("-" * 80)
    print(f"{'קובץ':<30} {'צמתים':<10} {'קשתות':<10} {'צפיפות':<12}")
    print("-" * 80)

    for result in valid_results:
        density = result['edges'] / (result['nodes'] * (result['nodes'] - 1) / 2) if result['nodes'] > 1 else 0
        print(f"{result['filename']:<30} {result['nodes']:<10} {result['edges']:<10} {density:<12.3f}")



# דוגמה לשימוש מהיר עם קבצים דחוסים
def quick_analysis(directory_path):
    """
    ניתוח מהיר של כל הגרפים (כולל .xz)
    """
    print("סריקה מהירה של גרפי PACE 2017 (כולל קבצים דחוסים)...")

    graph_info = []

    # חיפוש קבצי .gr ו-.gr.xz
    for pattern in ['*.gr', '*.gr.xz']:
        for filepath in Path(directory_path).rglob(pattern):
            nodes, edges = quick_file_scan(filepath)
            if nodes and edges:
                graph_info.append({
                    'file': filepath.name,
                    'nodes': nodes,
                    'edges': edges,
                    'compressed': str(filepath).endswith('.xz')
                })

    # מיון ותצוגה
    graph_info.sort(key=lambda x: x['nodes'])

    print(f"\nנמצאו {len(graph_info)} גרפים:")
    print("-" * 60)
    print(f"{'קובץ':<30} {'צמתים':<8} {'קשתות':<8} {'דחוס':<6}")
    print("-" * 60)
    for info in graph_info:
        compressed_mark = "כן" if info['compressed'] else "לא"
        print(f"{info['file']:<30} {info['nodes']:>6} {info['edges']:>8} {compressed_mark:<6}")



def batch_extract_info_with_xz():
    """
    דרך אולטרה מהירה באמצעות פקודות מערכת
    """
    print("חילוץ מידע מהיר מקבצי .xz...")

    import glob

    files = glob.glob("*.gr.xz") + glob.glob("**/*.gr.xz", recursive=True)

    results = []
    for filepath in files:
        try:
            # שימוש ב-xzcat לחילוץ מהיר של השורות הראשונות
            result = subprocess.run(['xzcat', filepath],
                                    capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.split('\n')[:20]  # רק 20 שורות ראשונות

                for line in lines:
                    if line.strip().startswith('p tw '):
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            nodes, edges = int(parts[2]), int(parts[3])
                            results.append({
                                'file': os.path.basename(filepath),
                                'nodes': nodes,
                                'edges': edges
                            })
                            break

        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"בעיה עם {filepath}")
            continue

    # הצגת תוצאות
    results.sort(key=lambda x: x['nodes'])
    print(f"\nנמצאו {len(results)} גרפים:")
    print("-" * 50)
    for info in results:
        print(f"{info['file']:<25} {info['nodes']:>6} צמתים, {info['edges']:>6} קשתות")


if __name__ == "__main__":
    # שינוי נתיב לתיקיית הגרפים שלך
    # שינוי נתיב לתיקיית הגרפים שלך
    pace_directory = "C:\\Users\\Osnat\\Dropbox\\named-graphs-master\\gr"

    if os.path.exists(pace_directory):
        # ניתוח מלא
        results = analyze_pace_directory(pace_directory)
        create_summary_report(results)
        export_to_csv(results)
    else:
        print("נתיב לא נמצא. אנא עדכן את המשתנה pace_directory")
        print("\nדוגמאות שימוש:")
        print("quick_analysis('/path/to/your/pace/directory')")
        print("batch_extract_info_with_xz()  # אם אתה בתיקיית הקבצים")