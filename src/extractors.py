import io
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from rapidocr import RapidOCR
from PIL import Image
from langchain_core.documents import Document

from .models import ParsedUnit

ocr_engine = RapidOCR()
def clean_text(text: str) -> str:
    text = text.replace('\x00', ' ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _doc(text: str, **metadata) -> ParsedUnit:
    text = clean_text(text)
    return ParsedUnit(text=text, metadata=metadata) if text else None


def extract_pdf(data: bytes, file_name: str) -> list[ParsedUnit]:
    import fitz
    units = []
    with fitz.open(stream=data, filetype='pdf') as pdf:
        if pdf.is_encrypted:
            raise ValueError('Encrypted/password-protected PDFs are not supported.')
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text('text')
            if clean_text(text):
                units.append(_doc(text, page=page_no, source_type='pdf'))
            else:
                pix = page.get_pixmap(matrix=fitz.Matrix(1.8, 1.8), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes('png')))

                result = ocr_engine(img)

                ocr_text = '\n'.join(result.txts) if result.txts else ''

                if clean_text(ocr_text):
                    units.append(
                        _doc(
                            ocr_text,
                            page=page_no,
                            source_type='pdf_ocr'
                        )
                    )
    return [u for u in units if u]


def extract_docx(data: bytes) -> list[ParsedUnit]:
    from docx import Document as DocxDocument
    d = DocxDocument(io.BytesIO(data))
    units = []
    for i, p in enumerate(d.paragraphs, start=1):
        if p.text.strip():
            units.append(_doc(p.text, paragraph=i, source_type='docx'))
    for ti, table in enumerate(d.tables, start=1):
        rows = [' | '.join(cell.text.strip() for cell in row.cells) for row in table.rows]
        units.append(_doc('\n'.join(rows), section=f'table {ti}', source_type='docx_table'))
    return [u for u in units if u]


def extract_xlsx(data: bytes, file_name: str) -> list[ParsedUnit]:
    book = pd.ExcelFile(io.BytesIO(data))
    units = []
    for sheet in book.sheet_names:
        df = pd.read_excel(book, sheet_name=sheet, dtype=str).fillna('')
        text = f'Sheet: {sheet}\n' + df.to_csv(index=False)
        units.append(_doc(text, sheet=sheet, source_type='xlsx'))
    return [u for u in units if u]


def extract_csv(data: bytes) -> list[ParsedUnit]:
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str).fillna('')
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(data), dtype=str, encoding='latin-1').fillna('')
    return [_doc(df.to_csv(index=False), section='CSV table', source_type='csv')]


def extract_pptx(data: bytes) -> list[ParsedUnit]:
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    units = []
    for slide_no, slide in enumerate(prs.slides, start=1):
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, 'text') and shape.text.strip():
                texts.append(shape.text)
        if texts:
            units.append(_doc('\n'.join(texts), slide=slide_no, source_type='pptx'))
    return [u for u in units if u]


def extract_json(data: bytes) -> list[ParsedUnit]:
    obj = json.loads(data.decode('utf-8-sig'))
    return [_doc(json.dumps(obj, ensure_ascii=False, indent=2), section='JSON document', source_type='json')]


def extract_xml(data: bytes) -> list[ParsedUnit]:
    from lxml import etree
    root = etree.fromstring(data)
    text = '\n'.join(t.strip() for t in root.itertext() if t.strip())
    return [_doc(text, section='XML document', source_type='xml')]


def extract_text(data: bytes, file_name: str) -> list[ParsedUnit]:
    text = data.decode('utf-8-sig', errors='strict')
    return [_doc(text, section='Text document', source_type='txt')]


def extract_image(data: bytes) -> list[ParsedUnit]:
    img = Image.open(io.BytesIO(data))
    img.verify()

    img = Image.open(io.BytesIO(data))

    result = ocr_engine(img)

    text = '\n'.join(result.txts) if result.txts else ''

    return [_doc(
        text,
        section='OCR image',
        source_type='image_ocr'
    )]


def extract_document(name: str, data: bytes) -> list[ParsedUnit]:
    ext = Path(name).suffix.lower()
    if ext == '.pdf': return extract_pdf(data, name)
    if ext == '.docx': return extract_docx(data)
    if ext == '.csv': return extract_csv(data)
    if ext == '.xlsx': return extract_xlsx(data, name)
    if ext == '.pptx': return extract_pptx(data)
    if ext == '.json': return extract_json(data)
    if ext == '.xml': return extract_xml(data)
    if ext == '.txt': return extract_text(data, name)
    if ext in {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}: return extract_image(data)
    if ext in {'.doc', '.xls', '.ppt'}:
        raise ValueError(f'Legacy {ext} files require LibreOffice/antiword conversion and are not enabled in this safe baseline.')
    raise ValueError(f'No extractor configured for {ext}.')


def units_to_documents(units: Iterable[ParsedUnit], file_name: str, file_id: str) -> list[Document]:
    docs = []
    for idx, unit in enumerate(units):
        metadata = {
            **unit.metadata,
            'file_name': file_name,
            'file_id': file_id,
            'unit_id': idx,
        }
        docs.append(Document(page_content=unit.text, metadata=metadata))
    return docs
