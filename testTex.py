import os

# Pfad zum Verzeichnis mit den Bildern und Textdateien
output_dir = 'doc/Output/'

# LaTeX-Dokumentkopf
latex_header = r"""
\documentclass{article}
\usepackage{graphicx}
\begin{document}
"""

# LaTeX-Dokumentfuß
latex_footer = r"""
\end{document}
"""

# \\begin{{{figure}}}[H]
#     \\centering
#     \\includegraphics[width=0.8\\textwidth]{{{image_path}}}
#     \\label{{{image_path}}}
# \\end{{{figure}}}
# Funktion zum Generieren des LaTeX-Codes für ein Bild und die zugehörige Textdatei
figure = "figure"
def generate_latex_for_image(image_path, text_path):
    with open(text_path, 'r') as file:
        text_content = file.read()
    
    # Entferne 'doc/' aus dem image_path
    relative_image_path = image_path.replace('doc/', '')
    
    basename_without_ext = os.path.splitext(os.path.basename(image_path))[0].replace('_', ' ')
    latex_code = f"""
\\subsection{{{basename_without_ext}}}
\\begin{{center}}
    \\includegraphics[width=0.8\\textwidth]{{{relative_image_path}}}
    \\label{{{basename_without_ext}}}
\\end{{center}}
\\begin{{verbatim}}
{text_content}
\\end{{verbatim}}
"""


    return latex_code

# LaTeX-Dokumentinhalt
latex_content = ""

# Durchlaufe alle Dateien im Verzeichnis
for filename in os.listdir(output_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(output_dir, filename)
        text_path = os.path.join(output_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        if os.path.exists(text_path):
            latex_content += generate_latex_for_image(image_path, text_path)

# Kombiniere den LaTeX-Dokumentkopf, -inhalt und -fuß
latex_document = latex_header + latex_content + latex_footer

# Schreibe das LaTeX-Dokument in eine Datei
with open('output.tex', 'w') as file:
    file.write(latex_document)

print("LaTeX-Dokument wurde erfolgreich generiert.")