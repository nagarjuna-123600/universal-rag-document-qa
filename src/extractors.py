import io
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image
from langchain_core.documents import Document

from .models import ParsedUnit


# --------------------------------------------------
# OCR
# --------------------------------------------------

# Lazy initialization:
# OCR is loaded only when an image is uploaded.
ocr_engine = None


def get_ocr_engine():
    global ocr_engine

    if ocr_engine is None:
        try:
            from rapidocr import RapidOCR

            ocr_engine = RapidOCR()

        except Exception as e:
            raise RuntimeError(
                f"OCR initialization failed: {e}"
            )

    return ocr_engine


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _doc(
    text: str,
    **metadata
) -> ParsedUnit | None:

    text = clean_text(text)

    return (
        ParsedUnit(
            text=text,
            metadata=metadata
        )
        if text
        else None
    )


# --------------------------------------------------
# PDF
# --------------------------------------------------

def extract_pdf(
    data: bytes,
    file_name: str
) -> list[ParsedUnit]:

    import fitz

    units = []

    with fitz.open(
        stream=data,
        filetype="pdf"
    ) as pdf:

        if pdf.is_encrypted:
            raise ValueError(
                "Encrypted/password-protected PDFs "
                "are not supported."
            )

        for page_no, page in enumerate(
            pdf,
            start=1
        ):

            text = page.get_text("text")

            if clean_text(text):

                units.append(
                    _doc(
                        text,
                        page=page_no,
                        source_type="pdf"
                    )
                )

            else:

                # No PDF OCR here.
                units.append(
                    _doc(
                        "This PDF page does not "
                        "contain extractable text.",
                        page=page_no,
                        source_type="pdf_no_text"
                    )
                )

    return [
        u for u in units
        if u
    ]


# --------------------------------------------------
# DOCX
# --------------------------------------------------

def extract_docx(
    data: bytes
) -> list[ParsedUnit]:

    from docx import Document as DocxDocument

    d = DocxDocument(
        io.BytesIO(data)
    )

    units = []

    # Paragraphs
    for i, p in enumerate(
        d.paragraphs,
        start=1
    ):

        if p.text.strip():

            units.append(
                _doc(
                    p.text,
                    paragraph=i,
                    source_type="docx"
                )
            )

    # Tables
    for ti, table in enumerate(
        d.tables,
        start=1
    ):

        rows = []

        for row in table.rows:

            rows.append(
                " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                )
            )

        units.append(
            _doc(
                "\n".join(rows),
                section=f"table {ti}",
                source_type="docx_table"
            )
        )

    return [
        u for u in units
        if u
    ]


# --------------------------------------------------
# XLSX
# --------------------------------------------------

def extract_xlsx(
    data: bytes,
    file_name: str
) -> list[ParsedUnit]:

    book = pd.ExcelFile(
        io.BytesIO(data)
    )

    units = []

    for sheet in book.sheet_names:

        df = pd.read_excel(
            book,
            sheet_name=sheet,
            dtype=str
        ).fillna("")

        text = (
            f"Sheet: {sheet}\n"
            + df.to_csv(index=False)
        )

        units.append(
            _doc(
                text,
                sheet=sheet,
                source_type="xlsx"
            )
        )

    return [
        u for u in units
        if u
    ]


# --------------------------------------------------
# CSV
# --------------------------------------------------

def extract_csv(
    data: bytes
) -> list[ParsedUnit]:

    try:

        df = pd.read_csv(
            io.BytesIO(data),
            dtype=str
        ).fillna("")

    except UnicodeDecodeError:

        df = pd.read_csv(
            io.BytesIO(data),
            dtype=str,
            encoding="latin-1"
        ).fillna("")

    return [
        _doc(
            df.to_csv(index=False),
            section="CSV table",
            source_type="csv"
        )
    ]


# --------------------------------------------------
# PPTX
# --------------------------------------------------

def extract_pptx(
    data: bytes
) -> list[ParsedUnit]:

    from pptx import Presentation

    prs = Presentation(
        io.BytesIO(data)
    )

    units = []

    for slide_no, slide in enumerate(
        prs.slides,
        start=1
    ):

        texts = []

        for shape in slide.shapes:

            if hasattr(shape, "text"):

                if shape.text.strip():

                    texts.append(
                        shape.text
                    )

        if texts:

            units.append(
                _doc(
                    "\n".join(texts),
                    slide=slide_no,
                    source_type="pptx"
                )
            )

    return [
        u for u in units
        if u
    ]


# --------------------------------------------------
# JSON
# --------------------------------------------------

def extract_json(
    data: bytes
) -> list[ParsedUnit]:

    obj = json.loads(
        data.decode("utf-8-sig")
    )

    return [
        _doc(
            json.dumps(
                obj,
                ensure_ascii=False,
                indent=2
            ),
            section="JSON document",
            source_type="json"
        )
    ]


# --------------------------------------------------
# XML
# --------------------------------------------------

def extract_xml(
    data: bytes
) -> list[ParsedUnit]:

    from lxml import etree

    root = etree.fromstring(data)

    text = "\n".join(
        t.strip()
        for t in root.itertext()
        if t.strip()
    )

    return [
        _doc(
            text,
            section="XML document",
            source_type="xml"
        )
    ]


# --------------------------------------------------
# TXT
# --------------------------------------------------

def extract_text(
    data: bytes,
    file_name: str
) -> list[ParsedUnit]:

    text = data.decode(
        "utf-8-sig",
        errors="strict"
    )

    return [
        _doc(
            text,
            section="Text document",
            source_type="txt"
        )
    ]


# --------------------------------------------------
# IMAGE OCR
# --------------------------------------------------

def extract_image(
    data: bytes
) -> list[ParsedUnit]:

    try:

        # Open image with Pillow
        img = Image.open(
            io.BytesIO(data)
        )

        # Validate image
        img.verify()

        # Re-open because verify() closes
        # the image internally
        img = Image.open(
            io.BytesIO(data)
        )

        # Convert unusual formats/modes
        # into RGB for OCR
        if img.mode not in (
            "RGB",
            "L"
        ):

            img = img.convert("RGB")

        # Get OCR engine
        ocr = get_ocr_engine()

        # RapidOCR accepts numpy arrays.
        # Import numpy only here so normal
        # document uploads do not initialize OCR.
        import numpy as np

        image_array = np.array(img)

        # Run OCR
        result = ocr(image_array)

        # RapidOCR result handling
        ocr_texts = []

        if result is not None:

            # Current RapidOCR result normally
            # exposes text through result.txts
            if hasattr(result, "txts"):

                if result.txts:

                    ocr_texts.extend(
                        result.txts
                    )

            # Compatibility with result objects
            # that expose txts differently
            elif isinstance(result, tuple):

                first = result[0]

                if first:

                    for item in first:

                        if (
                            isinstance(item, (list, tuple))
                            and len(item) >= 2
                        ):

                            text = item[1]

                            if text:
                                ocr_texts.append(
                                    str(text)
                                )

        text = "\n".join(
            str(t)
            for t in ocr_texts
            if str(t).strip()
        )

        text = clean_text(text)

        if not text:

            return [
                _doc(
                    "No readable text was detected "
                    "in this image.",
                    section="OCR image",
                    source_type="image_ocr"
                )
            ]

        return [
            _doc(
                text,
                section="OCR image",
                source_type="image_ocr"
            )
        ]

    except Exception as e:

        raise RuntimeError(
            f"Image OCR failed: {e}"
        )


# --------------------------------------------------
# MAIN DOCUMENT EXTRACTOR
# --------------------------------------------------

def extract_document(
    name: str,
    data: bytes
) -> list[ParsedUnit]:

    ext = Path(name).suffix.lower()

    if ext == ".pdf":
        return extract_pdf(data, name)

    if ext == ".docx":
        return extract_docx(data)

    if ext == ".csv":
        return extract_csv(data)

    if ext == ".xlsx":
        return extract_xlsx(data, name)

    if ext == ".pptx":
        return extract_pptx(data)

    if ext == ".json":
        return extract_json(data)

    if ext == ".xml":
        return extract_xml(data)

    if ext == ".txt":
        return extract_text(data, name)

    if ext in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".tif",
        ".tiff"
    }:

        return extract_image(data)

  if ext == ".doc":
    raise ValueError(
        "This is an older Microsoft Word (.doc) file. "
        "Please save it as .docx and upload the .docx file."
    )

if ext == ".xls":
    raise ValueError(
        "This is an older Microsoft Excel (.xls) file. "
        "Please save it as .xlsx and upload the .xlsx file."
    )

if ext == ".ppt":
    raise ValueError(
        "This is an older Microsoft PowerPoint (.ppt) file. "
        "Please save it as .pptx and upload the .pptx file."
    )

# --------------------------------------------------
# CONVERT TO LANGCHAIN DOCUMENTS
# --------------------------------------------------

def units_to_documents(
    units: Iterable[ParsedUnit],
    file_name: str,
    file_id: str
) -> list[Document]:

    docs = []

    for idx, unit in enumerate(units):

        metadata = {
            **unit.metadata,
            "file_name": file_name,
            "file_id": file_id,
            "unit_id": idx,
        }

        docs.append(
            Document(
                page_content=unit.text,
                metadata=metadata
            )
        )

    return docs
