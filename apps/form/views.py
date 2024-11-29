import json
import time
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import openai
from django import forms
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from pdf2image import convert_from_bytes
from pydantic import BaseModel, Field


client = OpenAI()


class ExtractForm(forms.Form):
    file = forms.FileField()


class Data(BaseModel):
    column_name: str
    value: str


class Row(BaseModel):
    data: list[Data]
    is_total: bool = Field(description="whether the row is a total or summary row")


class Table(BaseModel):
    full_heading: str = Field(
        description="the heading of the table prefixed with the heading of the tables parent sections"
    )
    rows: list[Row] = Field(
        description="the rows of the table with the column names included on each datum"
    )


class Tables(BaseModel):
    tables: list[Table]


class Holding(BaseModel):
    name: str
    cost_basis: float = Field(
        description="the cost basis of the holding, only match columns that are explicitly labeled as cost basis"
    )


class Holdings(BaseModel):
    holdings: list[Holding] = Field(
        description="the holdings in the account described by these tables, include individual holdings, do not include total or aggregate values, do not include holdings with incomplete data, only include tables that explicitly contain holdings"
    )


class InvestmentData(BaseModel):
    account_owner_name: str
    portfolio_value: float


def process_page(image_base64):
    return (
        client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at structured data extraction. You will be given a page from a pdf financial statement and should convert it into the given structure.  The structure is a list of tables, each with a full heading and a list of rows. Each row is a list of data points, each with a column name and a value.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        }
                    ],
                },
            ],
            response_format=Tables,
            temperature=0,
        )
        .choices[0]
        .message.parsed
    )


def extract_holdings(tables: Tables):
    for i in range(5):
        try:
            return (
                client.beta.chat.completions.parse(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at structured data extraction. You will be given a json document that contains a list of tables from a page of a financial statement and should convert it into the given structure. The output structure is a description of the specific individual holdings described by these tables if any.  Match column names closely, do not extrapolate or guess.",
                        },
                        {
                            "role": "user",
                            "content": tables.model_dump_json(),
                        },
                    ],
                    response_format=Holdings,
                    temperature=0,
                )
                .choices[0]
                .message.parsed
            )
        except openai.RateLimitError:
            time.sleep(i**2 * 5)


@csrf_exempt
def extract(request):
    if request.method == "POST":
        form = ExtractForm(request.POST, request.FILES)
        if not form.is_valid():
            return HttpResponse(status=400)
        file = form.cleaned_data["file"]
        images = convert_from_bytes(file.read())
        buffers = [BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG")
        images_base64 = (
            b64encode(buffer.getvalue()).decode("utf-8") for buffer in buffers
        )

        with ThreadPoolExecutor() as executor:
            table_sets = list(executor.map(process_page, images_base64))
            holdings_sets = list(executor.map(extract_holdings, table_sets))

        tables = [
            table.model_dump_json()
            for table_set in table_sets
            for table in table_set.tables
        ]

        all_tables_json = f"[{','.join(tables)}]"

        investment_data = (
            client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at structured data extraction. You will be given a json document that contains a list of tables from a financial statement and should convert it into the given structure. The output structure is a description of the investment information.",
                    },
                    {
                        "role": "user",
                        "content": all_tables_json,
                    },
                ],
                response_format=InvestmentData,
                temperature=0,
            )
            .choices[0]
            .message.parsed
        )

        response_data = {
            **investment_data.model_dump(),
            "holdings": [
                holding.model_dump()
                for holding_set in holdings_sets
                for holding in holding_set.holdings
            ],
        }
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json",
        )
    return HttpResponse(status=405)
