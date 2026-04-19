"""
Vercel Serverless Function
FastAPI uygulamasını Vercel'de çalıştırır
"""
import sys
from pathlib import Path

# app/ klasörünü Python path'e ekle
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.main import app as fastapi_app

# CORS kurulumu (Vercel domain'i için)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "*.vercel.app",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel handler
app = fastapi_app
