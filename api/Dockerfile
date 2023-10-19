FROM public.ecr.aws/lambda/python:3.11
COPY ./app ./app
COPY ./requirements.txt ./requirements.txt
RUN pip3 install -r ./requirements.txt
ENV TRANSFORMERS_CACHE=/tmp/
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/
CMD ["app.main.handler"]
