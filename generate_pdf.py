import os
import re
import subprocess
import markdown

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
md_path = os.path.join(BASE_DIR, "docs", "COMPLETE_APPLICATION_RUN_DOWN.md")
html_path = os.path.join(BASE_DIR, "docs", "COMPLETE_APPLICATION_RUN_DOWN.html")
pdf_path = os.path.join(BASE_DIR, "docs", "COMPLETE_APPLICATION_RUN_DOWN.pdf")

with open(md_path, "r", encoding="utf-8") as f:
    md_text = f.read()

# Protect LaTeX math expressions before markdown parsing
math_blocks = []
def replace_math_block(match):
    math_blocks.append(match.group(0))
    return f"MATHBLOCKPLACEHOLDER{len(math_blocks)-1}"

# Protect inline and block math
md_text_protected = re.sub(r'\$\$(.*?)\$\$', replace_math_block, md_text, flags=re.DOTALL)
md_text_protected = re.sub(r'\$(.*?)\$', replace_math_block, md_text_protected)

# Convert Markdown to HTML
html_body = markdown.markdown(
    md_text_protected,
    extensions=['tables', 'fenced_code', 'codehilite', 'toc']
)

# Restore math expressions
for i, block in enumerate(math_blocks):
    html_body = html_body.replace(f"MATHBLOCKPLACEHOLDER{i}", block)

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Complete Run Down - Multi-Agent Emergent Communication & Online Repair</title>
    <!-- MathJax for rendering math expressions -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{
        fontCache: 'global'
      }}
    }};
    </script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        @page {{
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #1e293b;
            background-color: #ffffff;
            line-height: 1.6;
            font-size: 10.5pt;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            font-size: 20pt;
            font-weight: 800;
            color: #0f172a;
            border-bottom: 2.5px solid #2563eb;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 16px;
        }}
        h2 {{
            font-size: 14pt;
            font-weight: 700;
            color: #1e3a8a;
            border-bottom: 1.5px solid #e2e8f0;
            padding-bottom: 6px;
            margin-top: 24px;
            margin-bottom: 12px;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 12pt;
            font-weight: 600;
            color: #334155;
            margin-top: 18px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }}
        h4 {{
            font-size: 11pt;
            font-weight: 600;
            color: #475569;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p, ul, ol {{
            margin-top: 0;
            margin-bottom: 10px;
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
            font-size: 9pt;
            border: 1px solid #e2e8f0;
        }}
        pre {{
            font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
            background-color: #0f172a;
            color: #f8fafc;
            padding: 12px 14px;
            border-radius: 6px;
            font-size: 8.5pt;
            overflow-x: auto;
            line-height: 1.45;
            page-break-inside: avoid;
            margin-top: 6px;
            margin-bottom: 14px;
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
            margin-top: 12px;
            margin-bottom: 16px;
            font-size: 9pt;
            page-break-inside: avoid;
        }}
        th, td {{
            padding: 7px 10px;
            border: 1px solid #cbd5e1;
            text-align: left;
        }}
        th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: #0f172a;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        blockquote {{
            margin: 12px 0;
            padding: 10px 14px;
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            color: #1e40af;
            font-size: 9.5pt;
            border-radius: 0 6px 6px 0;
            page-break-inside: avoid;
        }}
        blockquote p {{
            margin: 0;
        }}
        hr {{
            border: 0;
            height: 1px;
            background: #e2e8f0;
            margin: 20px 0;
        }}
        a {{
            color: #2563eb;
            text-decoration: none;
        }}
        .math-display {{
            overflow-x: auto;
            margin: 8px 0;
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

# Run headless Chrome / Edge to convert HTML to PDF
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
print(f"Return code: {res.returncode}")
if os.path.exists(pdf_path):
    print(f"SUCCESS: PDF created at {pdf_path} (size: {os.path.getsize(pdf_path)} bytes)")
else:
    print(f"Error: {res.stderr}")
