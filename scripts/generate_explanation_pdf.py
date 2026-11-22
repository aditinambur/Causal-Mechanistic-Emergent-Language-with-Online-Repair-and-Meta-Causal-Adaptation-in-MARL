import os
import subprocess
import markdown

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md_path = os.path.join(BASE_DIR, "docs", "PRESENTATION_RESULTS_EXPLANATION.md")
html_path = os.path.join(BASE_DIR, "docs", "PRESENTATION_RESULTS_EXPLANATION.html")
pdf_path = os.path.join(BASE_DIR, "docs", "PRESENTATION_RESULTS_EXPLANATION.pdf")

with open(md_path, "r", encoding="utf-8") as f:
    md_text = f.read()

# Convert Markdown to HTML with table and code extensions
html_body = markdown.markdown(
    md_text,
    extensions=['tables', 'fenced_code', 'toc']
)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Results Verification & Line-by-Line Breakdown</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        @page {{
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {{
                content: counter(page);
                font-family: 'Inter', sans-serif;
                font-size: 8pt;
                color: #64748b;
            }}
            @bottom-left {{
                content: "Multi-Agent Emergent Communication & Online Repair";
                font-family: 'Inter', sans-serif;
                font-size: 8pt;
                color: #64748b;
            }}
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #1e293b;
            background-color: #ffffff;
            line-height: 1.55;
            font-size: 10pt;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 18pt;
            font-weight: 800;
            color: #0f172a;
            border-bottom: 3px solid #2563eb;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 6px;
        }}
        h2 {{
            font-size: 13pt;
            font-weight: 700;
            color: #1e3a8a;
            border-bottom: 1.5px solid #e2e8f0;
            padding-bottom: 5px;
            margin-top: 22px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 11pt;
            font-weight: 600;
            color: #2563eb;
            margin-top: 16px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        h4 {{
            font-size: 10pt;
            font-weight: 600;
            color: #475569;
            margin-top: 12px;
            margin-bottom: 4px;
            page-break-after: avoid;
        }}
        p, ul, ol {{
            margin-top: 0;
            margin-bottom: 8px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        code {{
            font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 8.5pt;
            border: 1px solid #e2e8f0;
        }}
        pre {{
            font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
            background-color: #f8fafc;
            color: #0f172a;
            padding: 10px 12px;
            border-radius: 6px;
            font-size: 8.5pt;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #2563eb;
            overflow-x: auto;
            line-height: 1.45;
            page-break-inside: avoid;
            margin-top: 6px;
            margin-bottom: 12px;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
            border: none;
            font-size: inherit;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            margin-bottom: 14px;
            font-size: 8.5pt;
            page-break-inside: avoid;
        }}
        th, td {{
            padding: 6px 9px;
            border: 1px solid #cbd5e1;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            font-weight: 600;
            color: #0f172a;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        blockquote {{
            margin: 10px 0;
            padding: 8px 12px;
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            color: #1e40af;
            font-size: 9pt;
            border-radius: 0 4px 4px 0;
            page-break-inside: avoid;
        }}
        blockquote p {{
            margin: 0;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #e2e8f0;
            margin: 18px 0;
        }}
        strong {{
            color: #0f172a;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated HTML at {html_path}")

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
if not os.path.exists(chrome_path):
    chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

cmd = [
    chrome_path,
    "--headless=new",
    "--disable-gpu",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=5000",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    html_path
]

print(f"Running command: {' '.join(cmd)}")
res = subprocess.run(cmd, capture_output=True, text=True)
if os.path.exists(pdf_path):
    print(f"SUCCESS: PDF created at {pdf_path} (size: {os.path.getsize(pdf_path)} bytes)")
else:
    print(f"Error: {res.stderr}")
