import hashlib
import streamlit as st

from src.config import settings
from src.extractors import extract_document, units_to_documents
from src.rag import UniversalRAG
from src.security import UnsafeFileError, validate_upload


st.set_page_config(
    page_title='Universal RAG Document QA',
    page_icon='📚',
    layout='wide'
)


SUPPORTED = [
    'pdf', 'doc', 'docx', 'txt',
    'csv', 'xls', 'xlsx',
    'ppt', 'pptx',
    'json', 'xml',
    'jpg', 'jpeg', 'png',
    'webp', 'tiff', 'tif'
]


def get_rag():
    return UniversalRAG()


# Initialize RAG
if 'rag' not in st.session_state:
    try:
        st.session_state.rag = get_rag()
        st.session_state.file_names = {}
        st.session_state.messages = []
    except Exception as exc:
        st.error(str(exc))
        st.stop()


rag = st.session_state.rag


# -----------------------------
# Main UI
# -----------------------------

st.title('📚 Universal RAG Document QA')

st.caption(
    'Upload documents/images and ask questions grounded only in their content.'
)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.header('Upload & Sources')

    uploads = st.file_uploader(
        'Upload files',
        type=SUPPORTED,
        accept_multiple_files=True,
        max_upload_size=settings.max_file_mb,
        help=(
            f'Max {settings.max_file_mb} MB per file. '
            'APK/PFX and unsafe files are rejected.'
        ),
    )

    # ==================================================
    # PROCESS UPLOADED FILES
    # ==================================================

    if uploads:

        if len(uploads) > settings.max_files_per_upload:

            st.error(
                f'Maximum {settings.max_files_per_upload} '
                'files per upload batch.'
            )

        else:

            for uploaded in uploads:

                data = uploaded.getvalue()

                file_id = hashlib.sha256(
                    data
                ).hexdigest()

                # Only process new files
                if file_id not in rag.files:

                    try:

                        validate_upload(
                            uploaded.name,
                            data,
                            settings.max_file_mb
                        )

                        units = extract_document(
                            uploaded.name,
                            data
                        )

                        docs = units_to_documents(
                            units,
                            uploaded.name,
                            file_id
                        )

                        rag.add_file(
                            uploaded.name,
                            file_id,
                            docs
                        )

                        st.session_state.file_names[
                            file_id
                        ] = uploaded.name

                        st.success(
                            f'Indexed: {uploaded.name}'
                        )

                    except (
                        UnsafeFileError,
                        ValueError,
                        RuntimeError
                    ) as exc:

                        st.error(
                            f'{uploaded.name}: {exc}'
                        )

                    except Exception as exc:

                        st.error(
                            f'{uploaded.name}: '
                            f'processing failed ({exc})'
                        )

    # ==================================================
    # REMOVE FILES THAT WERE DELETED FROM UPLOADER
    # ==================================================

    active_file_ids = set()

    if uploads:

        for uploaded in uploads:

            data = uploaded.getvalue()

            file_id = hashlib.sha256(
                data
            ).hexdigest()

            active_file_ids.add(file_id)

    # Find files that are no longer in uploader
    deleted_file_ids = set(rag.files.keys()) - active_file_ids

    # Remove their FAISS indexes
    for file_id in deleted_file_ids:

        del rag.files[file_id]

        st.session_state.file_names.pop(
            file_id,
            None
        )

    # ==================================================
    # CLEAR CHAT
    # ==================================================

    if st.button(
        '🗑️ Clear Chat',
        use_container_width=True
    ):

        st.session_state['messages'] = []

        st.rerun()

    st.divider()

    st.markdown('**Security**')

    st.caption(
        'Files are validated before parsing. '
        'Uploaded content is treated as untrusted data.'
    )

    st.markdown('**RAG flow**')

    st.caption(
        'Validation → extraction/OCR → cleaning → '
        'chunks → embeddings → FAISS → retrieval → '
        'grounded LLM → sources'
    )

# -----------------------------
# Previous messages
# -----------------------------

for msg in st.session_state.get('messages', []):

    with st.chat_message(msg['role']):

        st.markdown(msg['content'])

        if msg.get('sources'):

            with st.expander('Sources'):

                for i, source in enumerate(
                    msg['sources'],
                    start=1
                ):

                    st.markdown(
                        f'**{i}. {source.file_name} — '
                        f'{source.location}**'
                    )

                    st.caption(
                        f'Chunk: {source.chunk_id} | '
                        f'Retrieval distance: '
                        f'{source.score:.4f}'
                    )

                    st.code(
                        source.text[:2000]
                    )


# -----------------------------
# Chat input
# -----------------------------

question = st.chat_input(
    'Ask a question about the uploaded files…'
)


if question:

    st.session_state['messages'].append({
        'role': 'user',
        'content': question
    })


    with st.chat_message('user'):
        st.markdown(question)


    with st.chat_message('assistant'):

        with st.spinner(
            'Retrieving relevant content…'
        ):

            try:

                # IMPORTANT:
                # No selected_file_ids anymore
                answer, sources, found = rag.answer(
                    question
                )

                st.markdown(answer)


                if found:

                    with st.expander(
                        'Sources',
                        expanded=True
                    ):

                        for i, source in enumerate(
                            sources,
                            start=1
                        ):

                            st.markdown(
                                f'**{i}. {source.file_name} — '
                                f'{source.location}**'
                            )

                            st.caption(
                                f'Chunk: {source.chunk_id} | '
                                f'Retrieval distance: '
                                f'{source.score:.4f}'
                            )

                            st.code(
                                source.text[:2000]
                            )


                st.session_state['messages'].append({
                    'role': 'assistant',
                    'content': answer,
                    'sources': sources
                })


            except Exception as exc:

                st.error(
                    f'Unable to answer the question: {exc}'
                )
