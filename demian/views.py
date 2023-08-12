import os
import json

# django module
from django.conf import settings
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from django.core.files.uploadedfile import UploadedFile
from .models import TextFileUpload
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from mysite.utils import rename
from common.forms import TextFileUploadForm

# django-rest-framework module
from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

# langchain module
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

# openai module
import openai

# pincone module
import pinecone

load_dotenv()
openai.api_key = "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"

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.environ.get("PINECONE_ENV"),  # next to api key in console
)


# pdf_pretreatments
def pdf_pretreatments(path):
    loader = PyPDFLoader(path)  # 유니코드 인코딩하지 않으면 불가능
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    # 이 텍스트 부분을 전처리 해야 함. 알고리즘으로 좀 더, flexible하게.
    texts = text_splitter.split_documents(documents)
    return texts


def pdf_pretreatments_dic_level(pdf_dic):
    loader = PyPDFDirectoryLoader(pdf_dic)  # 유니코드 인코딩하지 않으면 불가능
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )
    # 이 텍스트 부분을 전처리 해야 함. 알고리즘으로 좀 더, flexible하게.
    texts = text_splitter.split_documents(documents)
    return texts


# view function
chat_history = []
@api_view(['GET', 'POST'])
#@csrf_exempt
def post(request):    
    content = {}

    if request.method == "POST":
        if save_chain != []:
            chain = save_chain[-1]
            data = json.loads(request.body)
            message = data.get('message')
            question = message
            response = chain({"question": question, "chat_history": chat_history})
            answer = response["answer"]
            source = response["source_documents"]
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=answer))
            #print(chat_history)

            content['result'] = answer
            return Response({'message': answer})
        else:
            return Response({'message': "해석할 파일이 없습니다."})
    else:
        return Response({'message': "해석할 파일이 없습니다."})


save_chain = []
@csrf_exempt
def fileUpload(request):
    if request.method == "POST":
        file = request.FILES["file"]
        file_name = UploadedFile(file).name
        form = TextFileUploadForm(request.FILES)
        if form.is_valid():
            fileupload = TextFileUpload(text_file=file)
            fileupload.save()
            save_path = os.path.join(settings.MEDIA_ROOT, 'textuploads/', rename.pop())
            #savepath.append(save_path)
            texts = pdf_pretreatments(save_path)
            embeddings = OpenAIEmbeddings()
            index_name = "langchain-demo"
            if index_name not in pinecone.list_indexes():
    # we create a new index
                pinecone.create_index(
                    name=index_name,
                    metric='cosine',
                    dimension=1536  
            )
            vector_store = Pinecone.from_documents(texts,
                                     embeddings,
                                    index_name = index_name
                                     )
            #vector_store.persist()
            retriever = vector_store.as_retriever()
            model = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature="0",
                # verbose=True
            )
            chain = ConversationalRetrievalChain.from_llm(
            model,
            retriever=retriever,
            return_source_documents=True,
            # verbose=True,
            )
            save_chain.append(chain)
            return JsonResponse({'message': '파일 업로드가 성공적으로 완료되었습니다.'})
    else:
        return JsonResponse({'message': '파일을 업로드할 수 없습니다.'})


@login_required(login_url='common:login')
def index(request):
    return render(request, 'demian/index.html')