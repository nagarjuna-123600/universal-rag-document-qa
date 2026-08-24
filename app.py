import hashlib
import streamlit as st
from src.config import settings
from src.extractors import extract_document, units_to_documents
from src.rag import UniversalRAG
from src.security import UnsafeFileError, validate_upload

st.set_page_config(page_title='Universal RAG Document QA', page_icon='📚', layout='wide')

SUPPORTED = ['pdf','doc','docx','txt','csv','xls','xlsx','ppt','pptx','json','xml','jpg','jpeg','png','webp','tiff','tif']

@st.cache_resource(show_spinner=False)
def get_rag():
    return UniversalRAG()

if 'rag' not in st.session_state:
    try:
        st.session_state.rag = get_rag()
        st.session_state.file_names = {}
    except Exception as exc:
        st.error(str(exc))
        st.stop()

rag = st.session_state.rag

st.title('📚 Universal RAG Document QA')
st.caption('Upload documents/images, select the source files, and ask questions grounded only in their content.')

with st.sidebar:
    st.header('Upload & Sources')
    uploads = st.file_uploader(
        'Upload files',
        type=SUPPORTED,
        accept_multiple_files=True,
        max_upload_size=settings.max_file_mb,
        help=f'Max {settings.max_file_mb} MB per file. APK/PFX and unsafe files are rejected.',
    )
    if uploads:
        if len(uploads) > settings.max_files_per_upload:
            st.error(f'Maximum {settings.max_files_per_upload} files per upload batch.')
        else:
            for uploaded in uploads:
                data = uploaded.getvalue()
                file_id = hashlib.sha256(data).hexdigest()
                if file_id not in rag.files:
                    try:
                        validate_upload(uploaded.name, data, settings.max_file_mb)
                        units = extract_document(uploaded.name, data)
                        docs = units_to_documents(units, uploaded.name, file_id)
                        rag.add_file(uploaded.name, file_id, docs)
                        st.session_state.file_names[file_id] = uploaded.name
                        st.success(f'Indexed: {uploaded.name}')
                    except (UnsafeFileError, ValueError, RuntimeError) as exc:
                        st.error(f'{uploaded.name}: {exc}')
                    except Exception as exc:
                        st.error(f'{uploaded.name}: processing failed ({exc})')

    file_ids = list(rag.files.keys())
    selected = st.multiselect(
        'Search only these files',
        options=file_ids,
        default=file_ids,
        format_func=lambda x: rag.files[x].file_name,
    )
    st.divider()
    st.markdown('**Security**')
    st.caption('Files are validated before parsing. Uploaded content is treated as untrusted data.')
    st.markdown('**RAG flow**')
    st.caption('Validation → extraction/OCR → cleaning → chunks → embeddings → FAISS → retrieval → grounded LLM → sources')

if not selected:
    st.info('Upload at least one supported file and select it in the sidebar.')
    st.stop()

for msg in st.session_state.get('messages', []):
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg.get('sources'):
            with st.expander('Sources'):
                for i, source in enumerate(msg['sources'], start=1):
                    st.markdown(f'**{i}. {source.file_name} — {source.location}**')
                    st.caption(f'Chunk: {source.chunk_id} | Retrieval distance: {source.score:.4f}')
                    st.code(source.text[:2000])

question = st.chat_input('Ask a question about the selected files…')
if question:
    st.session_state.setdefault('messages', []).append({'role':'user','content':question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Retrieving relevant content…'):
            try:
                answer, sources, found = rag.answer(question, selected)
                st.markdown(answer)
                if found:
                    with st.expander('Sources', expanded=True):
                        for i, source in enumerate(sources, start=1):
                            st.markdown(f'**{i}. {source.file_name} — {source.location}**')
                            st.caption(f'Chunk: {source.chunk_id} | Retrieval distance: {source.score:.4f}')
                            st.code(source.text[:2000])
                st.session_state['messages'].append({'role':'assistant','content':answer,'sources':sources})
            except Exception as exc:
                st.error(f'Unable to answer the question: {exc}')
