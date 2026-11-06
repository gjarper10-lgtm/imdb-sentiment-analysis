"""Export this project's generated report to a readable PDF without Microsoft Word.

This covers the paragraphs, tables and inline figures used by build_report.py.
It is not a general-purpose DOCX renderer; the editable Word document remains available.
"""
from html import escape
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image


def build():
    root = Path(__file__).resolve().parent
    source = Document(root / 'Project_Report.docx')
    output = root / 'Project_Report.pdf'
    regular, bold = 'Times-Roman', 'Times-Bold'
    fonts = Path('C:/Windows/Fonts')
    if (fonts / 'times.ttf').exists():
        pdfmetrics.registerFont(TTFont('ProjectTimes', str(fonts / 'times.ttf')))
        pdfmetrics.registerFont(TTFont('ProjectTimesBold', str(fonts / 'timesbd.ttf')))
        regular, bold = 'ProjectTimes', 'ProjectTimesBold'
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('BodyProject', fontName=regular, fontSize=11, leading=15,
                              spaceAfter=7, alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle('TitleProject', fontName=bold, fontSize=17, leading=22,
                              alignment=TA_CENTER, spaceAfter=18))
    styles.add(ParagraphStyle('HeadingProject', fontName=bold, fontSize=14, leading=18,
                              spaceBefore=13, spaceAfter=8, keepWithNext=True, textColor=colors.HexColor('#1f3a5f')))
    styles.add(ParagraphStyle('SubheadingProject', parent=styles['HeadingProject'], fontSize=12, leading=16))
    styles.add(ParagraphStyle('CellProject', fontName=regular, fontSize=8, leading=10))
    width = A4[0] - 104
    story = []
    for element in source.element.body:
        if element.tag == qn('w:p'):
            p = WordParagraph(element, source._body)
            text = p.text.strip()
            if text and 'Right-click and choose Update Field' not in text and text != 'Table of Contents':
                name = p.style.name if p.style else ''
                style = styles['HeadingProject' if name == 'Heading 1' else
                               'SubheadingProject' if name.startswith('Heading') else 'BodyProject']
                if p.alignment == 1:
                    style = styles['TitleProject']
                prefix = '- ' if name.startswith('List') else ''
                story.append(Paragraph(escape(prefix + text), style))
            for blip in element.iter(qn('a:blip')):
                rid = blip.get(qn('r:embed'))
                if rid:
                    picture = Image(BytesIO(source.part.related_parts[rid].blob))
                    factor = min(width / picture.imageWidth, 390 / picture.imageHeight)
                    picture.drawWidth, picture.drawHeight = picture.imageWidth*factor, picture.imageHeight*factor
                    story.extend([Spacer(1, 8), picture, Spacer(1, 8)])
            for br in element.iter(qn('w:br')):
                if br.get(qn('w:type')) == 'page':
                    story.append(PageBreak())
        elif element.tag == qn('w:tbl'):
            table = WordTable(element, source._body)
            rows = [[Paragraph(escape(cell.text), styles['CellProject']) for cell in row.cells] for row in table.rows]
            ncols = len(rows[0])
            weights = [float(col.width or 1) for col in table.columns]
            col_widths = [width*w/sum(weights) for w in weights] if len(weights)==ncols else [width/ncols]*ncols
            rendered = Table(rows, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
            rendered.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbe5ef')),
                ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#afbdcc')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f5f8')])]))
            story.extend([rendered, Spacer(1, 10)])
    def footer(canvas, doc):
        canvas.setFont(regular, 9)
        canvas.drawString(52, 28, 'IMDB Comparative Sentiment Classification | Arnav Kumar | ICT-4442')
        canvas.drawRightString(A4[0]-52, 28, str(doc.page))
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=52, leftMargin=52,
                      topMargin=48, bottomMargin=48, title='IMDB Comparative Sentiment Classification',
                      author='Arnav Kumar').build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Saved {output}')


if __name__ == '__main__':
    build()
