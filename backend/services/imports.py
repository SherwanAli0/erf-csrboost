"""Bounded table selection for new uploads. Legacy endpoints stay unchanged. Author: Sherwan Ali."""
import gzip
import io
from pathlib import PurePosixPath
import zipfile

from .errors import ServiceError


def _refuse(message):
    raise ServiceError('APP_IMPORT_REFUSED',message)


def _choose(names,kind,selection):
    if len(names)>30 or any(len(n)>200 for n in names):
        _refuse('This container has too many or overly long table names. Export one table as CSV.')
    chosen=selection.get(kind)
    if chosen is not None and chosen not in names:
        _refuse('The chosen table is not in this upload. Choose one of its listed tables.')
    if chosen is None and len(names)>1:
        raise ServiceError('APP_IMPORT_SELECTION_REQUIRED','Choose a table from this upload, then upload and profile again.',
                           details={'selection_kind':kind,'choices':names})
    if not names:_refuse('No supported table was found in this container.')
    return chosen if chosen is not None else names[0]


def convert(raw,name,limits,engine,selection=None,depth=0):
    selection=selection or {}
    if not isinstance(selection,dict) or set(selection)-{'member','sheet'} or any(not isinstance(v,str) or len(v)>200 for v in selection.values()):
        _refuse('Import selection must name a sheet or archive member.')
    if depth>2:_refuse('Nested containers are limited to two levels. Extract the chosen table and upload it.')
    ext=PurePosixPath(name.lower()).suffix
    is_excel=ext in ('.xlsx','.xlsm','.xls') or raw[:8]==b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'
    # XLSX is a ZIP too. Bound its XML expansion before the spreadsheet reader.
    if raw.startswith(b'PK'):
        with zipfile.ZipFile(io.BytesIO(raw)) as book:
            infos=book.infolist()
            if len(infos)>1000 or sum(i.file_size for i in infos)>min(32*1024*1024,limits.max_bytes*4):
                _refuse('The archive expands beyond the import budget. Upload a smaller table.')
            for item in infos:
                path=PurePosixPath(item.filename.replace('\\','/'))
                if path.is_absolute() or '..' in path.parts or (path.parts and ':' in path.parts[0]):
                    _refuse('Unsafe archive paths are refused.')
            if '[Content_Types].xml' in book.namelist():is_excel=True
            if not is_excel:
                allowed={'.csv','.tsv','.txt','.data','.dat','.xlsx','.xls','.json','.jsonl','.dta','.arff'}
                names=[i.filename for i in infos if not i.is_dir() and PurePosixPath(i.filename.lower()).suffix in allowed and not i.filename.startswith('__MACOSX/')]
                if len(set(names))!=len(names):_refuse('Duplicate archive member names are refused.')
                chosen=_choose(names,'member',selection)
                if book.getinfo(chosen).file_size>limits.max_bytes:_refuse('The selected table exceeds the decompressed byte limit.')
                with book.open(chosen) as f:inner=f.read(limits.max_bytes+1)
                if len(inner)>limits.max_bytes:_refuse('The selected table exceeds the decompressed byte limit.')
                text,notes=convert(inner,chosen,limits,engine,{'sheet':selection['sheet']} if 'sheet' in selection else {},depth+1)
                return text,['Selected archive member: '+chosen]+notes
    if raw.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:inner=f.read(limits.max_bytes+1)
        if len(inner)>limits.max_bytes:_refuse('GZIP expansion exceeds the decompressed byte limit.')
        text,notes=convert(inner,name[:-3] if name.lower().endswith('.gz') else name,limits,engine,selection,depth+1)
        return text,['Bounded GZIP decompression.']+notes
    if is_excel:
        import pandas as pd
        with pd.ExcelFile(io.BytesIO(raw)) as book:
            sheet=_choose(book.sheet_names,'sheet',selection)
            frame=book.parse(sheet_name=sheet,nrows=limits.max_rows+2)
        frame=frame.dropna(how='all').dropna(axis=1,how='all')
        if len(frame)>limits.max_rows:_refuse('The selected sheet exceeds the row limit.')
        if frame.empty:_refuse('The selected sheet is empty. Choose a table with rows and columns.')
        return engine._promote_headers(frame).to_csv(index=False),['Selected worksheet: '+sheet]
    if selection:_refuse('A sheet/member selection was supplied for a plain table.')
    return engine.any_to_csv(raw,name)
