import requests
from bs4 import BeautifulSoup
from db import Product, engine, Session

def parse_product_row(row):
    return {
        "name": row[0],
        "calories": (row[1]),
        "protein": (row[2]),
        "fats": (row[3]),
        "carbohydrates": (row[4]),
    }


def parse_from_site(general_url, pages, headers):
    for i in pages:
        uurl = general_url + str(i) + "/"
        response = requests.get(uurl)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            header = (soup.find('h1')).text


            if header not in headers:
                one_row = []
                for el in soup.find_all('td'):
                    if "uk-text-right" in (el.get("class") or []):
                        nutritional_value = el.text.strip()
                        one_row.append(nutritional_value)

                    elif el.find("a"):
                        name_of_product = el.text.strip()
                        one_row.append(name_of_product)

                    if len(one_row) >= 5:
                        with Session(engine) as session:
                            data = parse_product_row(one_row)
                            product = Product(**data)
                            session.add(product)
                            session.commit()

                        one_row = []

                headers.append(header)

    return headers

all_indexes = [21214, 21215, 21217, 21218, 21219, 21241, 21242, 21243, 21244, 21245, 21247, 21248, 21249, 21250, 21251, 21252, 21253, 21254, 21273, 21274, 21327, 21484, 21545, 21546, 21636, 22471, 22655, 23901, 23977, 24127, 24501, 24502, 24503, 24504, 24506, 24507, 24508, 24509, 24511, 24512, 24513, 24514, 24515, 24516, 24517, 24518, 24519, 24520, 24522, 24523, 24524, 24525, 24526, 24527, 24528, 24529, 24555, 24556, 25675]
# Продукты питания свежие и приготовленные
url = f'https://health-diet.ru/base_of_food/food_'
all_products = parse_from_site(url, all_indexes, [])


# Готовые блюда и рецепты
all_indexes = [21214, 21215, 21217, 21218, 21219, 21241, 21242, 21243, 21244, 21245, 21247, 21248, 21249, 21250, 21251, 21252, 21253, 21254, 21273, 21274, 21327, 21484, 21545, 21546, 21636, 22471, 22655, 23901, 23977, 24127, 24501, 24502, 24503, 24504, 24506, 24507, 24508, 24509, 24511, 24512, 24513, 24514, 24515, 24516, 24517, 24518, 24519, 24520, 24522, 24523, 24524, 24525, 24526, 24527, 24528, 24529, 24555, 24556, 25675]
url = f'https://health-diet.ru/base_of_meals/meals_'
all_meals = parse_from_site(url, all_indexes, all_products)


# Продукты по брендам и производителям
all_indexes = [109145, 107024, 120802, 108930, 103503, 120566, 111530, 100481, 102398, 120310, 124912, 121338, 114851, 124180, 117741]
url = f'https://health-diet.ru/base_of_food/food_/'
all_brands = parse_from_site(url, all_indexes, all_meals)



