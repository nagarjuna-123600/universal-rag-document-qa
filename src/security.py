import hashlib
import io
import os
import zipfile
from pathlib import Path

ALLOWED_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.txt', '.csv', '.xls', '.xlsx', '.ppt', '.pptx',
    '.json', '.xml', '.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif'
}
EXPLICITLY_BLOCKED = {'.apk', '.pfx', '.exe', '.dll', '.msi', '.bat', '.cmd', '.ps1', '.sh'}

MAGIC = {
    b'%PDF-': 'pdf',
    b'PK\x03\x04': 'zip',
    b'\x89PNG\r\n\x1a\n': 'png',
    b'\xff\xd8\xff': 'jpeg',
    b'RIFF': 'riff',
    b'II*\x00': 'tiff',
    b'MM\x00*': 'tiff',
    b'MZ': 'executable',
}

class UnsafeFileError(ValueError):
    pass

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _magic(data: bytes) -> str:
    for sig, kind in MAGIC.items():
        if data.startswith(sig):
            return kind
    return 'unknown'

def _validate_zip(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            if zf.testzip() is not None:
                raise UnsafeFileError('Corrupted archive detected.')
            infos = zf.infolist()
            if len(infos) > 2000:
                raise UnsafeFileError('Archive contains too many entries.')
            total_uncompressed = 0
            for info in infos:
                name = info.filename.replace('\\', '/')
                if name.startswith('/') or '..' in Path(name).parts:
                    raise UnsafeFileError('Archive contains an unsafe path.')
                if info.flag_bits & 0x1:
                    raise UnsafeFileError('Encrypted archives are not supported.')
                total_uncompressed += info.file_size
                if info.file_size > 100 * 1024 * 1024:
                    raise UnsafeFileError('Archive entry is too large.')
                if info.compress_size and info.file_size / info.compress_size > 1000:
                    raise UnsafeFileError('Suspicious compression ratio detected.')
            if total_uncompressed > 300 * 1024 * 1024:
                raise UnsafeFileError('Archive expands beyond the safety limit.')
    except zipfile.BadZipFile as exc:
        raise UnsafeFileError('Invalid or corrupted ZIP-based document.') from exc

def validate_upload(name: str, data: bytes, max_mb: int) -> dict:
    ext = Path(name).suffix.lower()
    if ext in EXPLICITLY_BLOCKED:
        raise UnsafeFileError(f'{ext} files are not allowed.')
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsafeFileError(f'Unsupported file type: {ext or "unknown"}.')
    if not data:
        raise UnsafeFileError('Empty files are not allowed.')
    if len(data) > max_mb * 1024 * 1024:
        raise UnsafeFileError(f'File exceeds the {max_mb} MB limit.')

    kind = _magic(data[:16])
    if kind == 'executable':
        raise UnsafeFileError('Executable content is not allowed.')
    if ext == '.pdf' and kind != 'pdf':
        raise UnsafeFileError('The file extension does not match a PDF signature.')
    if ext in {'.png'} and kind != 'png':
        raise UnsafeFileError('The file is not a valid PNG.')
    if ext in {'.jpg', '.jpeg'} and kind != 'jpeg':
        raise UnsafeFileError('The file is not a valid JPEG.')
    if ext in {'.tiff', '.tif'} and kind != 'tiff':
        raise UnsafeFileError('The file is not a valid TIFF.')
    if ext in {'.docx', '.xlsx', '.pptx'}:
        if kind != 'zip':
            raise UnsafeFileError('The Office Open XML file is invalid.')
        _validate_zip(data)
    if ext in {'.doc', '.xls', '.ppt'}:
        # Legacy OLE files commonly start with this signature.
        if not data.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
            raise UnsafeFileError('The legacy Office file signature is invalid.')
    if ext == '.apk':
        raise UnsafeFileError('.apk files are never accepted.')
    if ext == '.pfx':
        raise UnsafeFileError('.pfx files are never accepted.')

    return {'extension': ext, 'sha256': sha256_bytes(data), 'magic': kind}
