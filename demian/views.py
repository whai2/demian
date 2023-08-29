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
from django.core.files.storage import FileSystemStorage

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
openai.api_key = os.environ.get("OPENAI_API_KEY")

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
chat_history_db = {}
#chat_history = []
chain_db = {}
@api_view(['GET', 'POST'])
#@csrf_exempt
def post(request):    
    content = {}
    try:
        if request.method == "POST":
            data = json.loads(request.body)
            uid = data.get('uid')
            message = data.get('message')
            if f'{uid}' in chain_db:
                chain = chain_db[f'{uid}'][-1]
                if f'{uid}' not in chat_history_db:
                    chat_history = []
                    chat_history_db[f'{uid}'] = chat_history
                question = message
                response = chain({"question": question, "chat_history": chat_history_db[f'{uid}']})
                answer = response["answer"]
                source = response["source_documents"]
                chat_history_db[f'{uid}'].append(HumanMessage(content=question))
                chat_history_db[f'{uid}'].append(AIMessage(content=answer))
                page = int(source[0].metadata['page'])
                page = page+1
                print(page)
                page_content = source[0].page_content
                print(page_content)

                content['result'] = answer
                return Response({'message': answer,'page':f'참고 페이지: {page} 페이지','page_content':f'참고 내용: {page_content}'})
            
            else:
                return Response({'message': "해석할 파일이 없습니다.",'page':"참고 페이지: 없음","page_content":"참고 내용: 없음" })
            
    except Exception as e:
        return JsonResponse({'message': '오류가 발생했습니다.','page':"참고 페이지: 없음","page_content":"참고 내용: 없음" })


@csrf_exempt
def fileUpload(request):
    if request.method == "POST":
        try:
            allowed_ext = ['pdf']

            file = request.FILES["file"]
            uid = request.POST['uid']
            ext = str(request.FILES['file']).split('.')[-1].lower()

            if not ext in allowed_ext:
                return JsonResponse({'messages': '허용된 확장자가 아닙니다.(가능한 확장자: pdf)','page':"참고 페이지: 없음","page_content":"참고 내용: 없음" })
            
            upload_path = os.path.join(settings.MEDIA_ROOT,'flashuploads/')
                        
            fs = FileSystemStorage(location=upload_path)
            filename = fs.save(f'flashfile.{ext}', file)
            save_path = os.path.join(upload_path, f'flashfile.{ext}')
            
            texts = pdf_pretreatments(save_path)

            #지우기
            fs.delete(f'flashfile.{ext}')

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
            retriever = vector_store.as_retriever()
            model = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature="0",
                         verbose=True
            )
            chain = ConversationalRetrievalChain.from_llm(
                    model,
                    retriever=retriever,
                    return_source_documents=True,
                     verbose=True,
            )
            

            empty_list = []
            empty_list.append(chain)
            chain_db[f'{uid}'] = empty_list
            
            return JsonResponse({'message': '파일 업로드가 성공적으로 완료되었습니다.','page':"참고 페이지: 없음","page_content":"참고 내용: 없음" })
        
        except Exception as e:
            return JsonResponse({'message': '파일을 업로드할 수 없습니다.','page':"참고 페이지: 없음","page_content":"참고 내용: 없음"})


@login_required(login_url='common:login')
def index(request):
    return render(request, 'demian/index.html')