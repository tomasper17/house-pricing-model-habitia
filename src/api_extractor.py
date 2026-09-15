"""
Idealista API client for extracting property data.
Handles OAuth2 authentication and property search requests.
"""

import requests
import base64
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import os

from src import rutas


class IdeallistaAPIClient:
    """Client for Idealista API with OAuth2 authentication."""

    BASE_URL = "https://api.idealista.com"
    TOKEN_ENDPOINT = f"{BASE_URL}/oauth/token"
    SEARCH_ENDPOINT = f"{BASE_URL}/3.5/{{country}}/search"

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize API client with credentials.

        Args:
            api_key: Your Idealista API key
            api_secret: Your Idealista API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = None
        self.token_expires_at = None

    def _encode_credentials(self) -> str:
        """Encode API key:secret in Base64 format."""
        credentials = f"{self.api_key}:{self.api_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return encoded

    def authenticate(self) -> bool:
        """
        Request OAuth2 bearer token.

        Returns:
            True if successful, False otherwise
        """
        headers = {
            "Authorization": f"Basic {self._encode_credentials()}",
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
        }

        data = {
            "grant_type": "client_credentials",
            "scope": "read"
        }

        try:
            response = requests.post(self.TOKEN_ENDPOINT, headers=headers, data=data)
            response.raise_for_status()

            result = response.json()
            self.access_token = result["access_token"]
            self.token_expires_at = time.time() + result["expires_in"] - 60  # 60s buffer

            print(f"✓ Authentication successful. Token expires in {result['expires_in']}s")
            return True

        except requests.exceptions.RequestException as e:
            print(f"✗ Authentication failed: {e}")
            return False

    def _ensure_token_valid(self) -> bool:
        """Refresh token if expired."""
        if not self.access_token or time.time() >= self.token_expires_at:
            return self.authenticate()
        return True

    def search_properties(
        self,
        country: str = "es",
        operation: str = "rent",
        property_type: str = "homes",
        location_id: Optional[str] = None,
        center: Optional[str] = None,
        distance: Optional[int] = None,
        max_price: Optional[int] = None,
        min_price: Optional[int] = None,
        bedrooms: Optional[str] = None,
        max_items_per_page: int = 50,
        max_pages: Optional[int] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Search for properties using Idealista API.

        Args:
            country: 'es', 'it', or 'pt'
            operation: 'sale' or 'rent'
            property_type: 'homes', 'offices', 'premises', 'garages', 'bedrooms'
            location_id: Idealista location code (use either this or center+distance)
            center: Coordinates as "lat,lon" (requires distance)
            distance: Distance in meters from center
            max_price: Maximum price filter
            min_price: Minimum price filter
            bedrooms: Comma-separated list (e.g., "1,2,3" or "4" for 4+)
            max_items_per_page: Items per page (max 50)
            max_pages: Maximum pages to fetch (None = all)
            **kwargs: Additional filters (e.g., numPage for starting page)

        Returns:
            List of property dictionaries
        """
        if not self._ensure_token_valid():
            return []

        # Validate inputs
        if not location_id and not (center and distance):
            print("Error: Must specify either location_id or (center + distance)")
            return []

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

        url = self.SEARCH_ENDPOINT.format(country=country)

        all_properties = []
        page_num = kwargs.get("numPage", 1)
        pages_fetched = 0

        while True:
            params = {
                "operation": operation,
                "propertyType": property_type,
                "maxItems": max_items_per_page,
                "numPage": page_num,
                "locale": "es"
            }

            # Add optional filters
            if location_id:
                params["locationId"] = location_id
            if center:
                params["center"] = center
            if distance:
                params["distance"] = distance
            if max_price:
                params["maxPrice"] = max_price
            if min_price:
                params["minPrice"] = min_price
            if bedrooms:
                params["bedrooms"] = bedrooms

            # Add any additional kwargs
            for key, value in kwargs.items():
                if key not in ["numPage"]:
                    params[key] = value

            try:
                print(f"Fetching page {page_num}...", end=" ")
                response = requests.post(url, headers=headers, data=params)
                response.raise_for_status()

                data = response.json()
                properties = data.get("elementList", [])
                total = data.get("total", 0)
                total_pages = data.get("totalPages", 0)
                self.last_total_pages = total_pages

                print(f"✓ Got {len(properties)} items (total: {total})")

                all_properties.extend(properties)
                pages_fetched += 1

                # Stop conditions
                if not properties or (max_pages and pages_fetched >= max_pages):
                    break

                if page_num >= total_pages:
                    break

                page_num += 1
                time.sleep(0.5)  # Rate limiting

            except requests.exceptions.RequestException as e:
                print(f"✗ Request failed: {e}")
                break

        print(f"\nTotal properties fetched: {len(all_properties)}")
        return all_properties

    def search_random_pages(self, n_pages: int, seed: int = 0, **search) -> List[Dict]:
        """
        Random sample of result pages: requests page 1 to learn the number of pages, then
        `n_pages` distinct page numbers drawn with `seed` among all pages. Costs n_pages + 1
        requests (n_pages if page 1 is drawn). Use a stable `order` (e.g. publicationDate) so
        pages do not reshuffle between requests. Chosen pages are kept in `self.last_pages`.
        """
        search.pop("max_pages", None)
        search.pop("numPage", None)
        first = self.search_properties(max_pages=1, numPage=1, **search)
        total_pages = getattr(self, "last_total_pages", 0)
        pages = sorted(random.Random(seed).sample(range(1, total_pages + 1), min(n_pages, total_pages)))
        self.last_pages = pages
        print(f"Random pages among {total_pages}: {pages}")

        properties = list(first) if 1 in pages else []
        for page in pages:
            if page == 1:
                continue
            time.sleep(0.5)  # Rate limiting
            properties.extend(self.search_properties(max_pages=1, numPage=page, **search))
        return properties

    def save_to_csv(self, properties: List[Dict], filename: str = None) -> str:
        """
        Save properties to CSV file.

        Args:
            properties: List of property dictionaries
            filename: Output filename (default: data_extraction_{timestamp}.csv)

        Returns:
            Path to saved file
        """
        if not properties:
            print("No properties to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_extraction_{timestamp}.csv"

        # Flatten nested structures
        df = pd.json_normalize(properties)

        # Create data/api_idealista directory if it doesn't exist
        output_dir = str(rutas.API_IDEALISTA)
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        print(f"✓ Saved {len(properties)} properties to {filepath}")
        return filepath

    def save_to_json(self, properties: List[Dict], filename: str = None) -> str:
        """
        Save properties to JSON file.

        Args:
            properties: List of property dictionaries
            filename: Output filename

        Returns:
            Path to saved file
        """
        if not properties:
            print("No properties to save")
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_extraction_{timestamp}.json"

        output_dir = str(rutas.API_IDEALISTA)
        os.makedirs(output_dir, exist_ok=True)

        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(properties, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(properties)} properties to {filepath}")
        return filepath


def main():
    """Example usage."""
    api_key = os.environ.get("IDEALISTA_API_KEY")
    api_secret = os.environ.get("IDEALISTA_API_SECRET")

    if not api_key or not api_secret:
        print("Error: Set IDEALISTA_API_KEY and IDEALISTA_API_SECRET environment variables")
        return

    # Initialize client
    client = IdeallistaAPIClient(api_key, api_secret)

    # Authenticate
    if not client.authenticate():
        return

    # Madrid ciudad (municipio 28079); "0-EU-ES-28" es la provincia entera
    properties = client.search_properties(
        country="es",
        operation="sale",
        property_type="homes",
        location_id="0-EU-ES-28-07-001-079",
        min_price=200000,
        max_price=300000,
        max_pages=5  # Limit to 5 pages for testing
    )

    # Save results
    if properties:
        client.save_to_csv(properties, "madrid_sales.csv")
        client.save_to_json(properties, "madrid_sales.json")


if __name__ == "__main__":
    main()
