"""Tests E2E Selenium.

Prérequis : l'API tourne (uvicorn ou docker compose) et Google Chrome est installé.
Exécution :  pytest -m e2e
URL cible modifiable :  BASE_URL=http://SERVER_IP:8000 pytest -m e2e
"""
import os
import shutil

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

pytestmark = pytest.mark.e2e


@pytest.fixture
def driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # chromedriver local : évite le téléchargement par Selenium Manager
    chromedriver = os.getenv("CHROMEDRIVER") or shutil.which("chromedriver")
    service = Service(executable_path=chromedriver) if chromedriver else Service()
    drv = webdriver.Chrome(options=options, service=service)
    yield drv
    drv.quit()


def test_swagger_docs(driver):
    driver.get(f"{BASE_URL}/docs")
    # Le titre de Swagger est "<titre de l'app> - Swagger UI"
    assert "Swagger UI" in driver.title


def test_prediction_from_ui(driver):
    driver.get(f"{BASE_URL}/ui/")

    champ = driver.find_element(By.ID, "nb_reclamations")
    champ.clear()
    champ.send_keys("3")

    driver.find_element(By.ID, "predict").click()

    WebDriverWait(driver, 10).until(
        EC.text_to_be_present_in_element((By.ID, "result"), "Résultat")
    )
    assert "CHURN" in driver.find_element(By.ID, "result").text
