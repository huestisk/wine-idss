from bs4 import BeautifulSoup
import os
import time
import requests
import re
import json
import glob
import pickle
import threading

BASE_URL = "https://www.winemag.com/?s=&drink_type=wine&pub_date_web={0}&page={1}"

session = requests.Session()
HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36"
    )
}

FILENAME = "data/winemag-data"
URLFILENAME = "data/review_urls.pkl"

UNKNOWN_FORMAT = 0
APPELLATION_FORMAT_0 = 1
APPELLATION_FORMAT_1 = 2
APPELLATION_FORMAT_2 = 3


class Scraper:
    """Scraper for Winemag.com to collect wine reviews"""

    def __init__(self, pages_to_scrape=(1, 1), years_to_scrape=[], clear_old_data=False):
        self.pages_to_scrape = pages_to_scrape
        self.years_to_scrape = years_to_scrape
        self.clear_old_data = clear_old_data
        self.session = requests.Session()
        self.start_time = time.time()
        self.reviews = []
        
    def get_response(self, url, retry=True):
        try:
            return self.session.get(url, headers=HEADERS)
        except requests.exceptions.ConnectionError:
            if retry:
                self.save_combined_data()
                print("Connection refused: waiting for 30 mins...")
                time.sleep(30*60) # wait for 10 mins
                self.session = requests.Session()
                self.get_response(url, retry=False)
            else:
                raise Exception("Connection refused")
        except:
            if retry:
                self.save_combined_data()
                print("Timeout: waiting for 10 mins...")
                time.sleep(10*60) # wait for 10 mins
                self.session = requests.Session()
                self.get_response(url, retry=False)
            else:
                raise Exception("Timeout: something went wrong...")

    def scrape_site_for_urls(self, prev_reviews=set()):
        self.all_review_urls = prev_reviews
        for i, year in enumerate(self.years_to_scrape):
            pages_to_scrape = (self.pages_to_scrape[2*i], self.pages_to_scrape[2*i+1])
            for page in range(pages_to_scrape[0], pages_to_scrape[1]):
                response = self.get_response(BASE_URL.format(year,page))
                threading.Thread(target=self.scrape_thread_urls, args=[response, page]).start()
                # self.scrape_thread_urls(response, page)
            
            time.sleep(15)
            print("Finished with year " + str(year) + "!")
            with open(URLFILENAME, "wb") as f:
                pickle.dump(self.all_review_urls, f)

        return self.all_review_urls            

    def scrape_thread_urls(self, response, page):
        soup = BeautifulSoup(response.content, "html.parser")
        reviews = soup.find_all("li", {"class": "review-item"})

        nulls = 0
        for review in reviews:
            try:
                self.all_review_urls.add(
                    review.find("a", {"class": "review-listing"})["href"]
                )
            except:
                nulls += 1

        if nulls > 10:
            with open(URLFILENAME, "wb") as f:
                pickle.dump(self.all_review_urls, f)
            raise Exception("No more urls are being parsed...")

        if page==1 or page % 100 == 0:
            print(
                "Scraped: page {2} | {0} urls | {1} sec elapsed\r".format(
                    len(self.all_review_urls),
                    round(time.time() - self.start_time, 2),
                    page
                )
            )

    def scrape_reviews(self, review_urls, retry_count=0):
        # Add files from outputfile
        try:
            with open("{}.json".format(FILENAME), "r") as fin:
                self.reviews = json.load(fin)
        except FileNotFoundError:
            pass
        
        revs_loaded = len(self.reviews)
        print(str(revs_loaded) + " reviews were loaded.")

        for idx, review_url in enumerate(review_urls):
            # FIXME: first 46,5k saved in separate file
            if idx <= 46500 + revs_loaded or review_url[-2:] == "//": 
                continue
            response = self.get_response(review_url)
            threading.Thread(target=self.scrape_thread, args=[response, idx]).start()

            if idx % 500 == 0:
                self.save_combined_data()

        self.save_combined_data()

    def scrape_thread(self, review, idx):
        review_soup = BeautifulSoup(review.content, "html.parser")
        try:
            scrape_data = self.parse_review(review_soup)
        except Exception as e:
            print("Encountered error", e)
            return
        self.reviews.append(scrape_data)
        if idx % 100 == 0:
            print("Reviews done {0} in {1} secs\r".format(idx,
                round(time.time() - self.start_time, 2)))

    def save_combined_data(self):
        filename = "{}.json".format(FILENAME)
        try:
            os.replace(filename, filename + '.bak')
        except:
            self.clear_output_data()
        
        with open(filename, "w") as fout:
            json.dump(self.reviews, fout)
        print("Saved data with " + str(len(self.reviews)) + " reviews.")

    def parse_review(self, review_soup):
        review_format = self.determine_review_format(review_soup)
        points = review_soup.find("span", {"id": "points"}).contents[0]
        title = review_soup.title.string.split(" Rating")[0]
        description = review_soup.find("p", {"class": "description"}).contents[0]

        info_containers = review_soup.find("ul", {"class": "primary-info"}).find_all(
            "li", {"class": "row"}
        )

        if review_format["price_index"] is not None:
            try:
                price_string = (
                    info_containers[review_format["price_index"]]
                    .find("div", {"class": "info"})
                    .span.span.contents[0]
                    .split(",")[0]
                )
            except:
                raise ReviewFormatException("Unexpected price format")
            # Sometimes price is N/A
            try:
                price = int(re.sub("[$]", "", price_string))
            except ValueError:
                price = None
        else:
            price = None

        if review_format["designation_index"] is not None:
            try:
                designation = (
                    info_containers[review_format["designation_index"]]
                    .find("div", {"class": "info"})
                    .span.span.contents[0]
                )
            except:
                raise ReviewFormatException("Unexpected designation format")
        else:
            designation = None

        if review_format["variety_index"] is not None:
            try:
                variety = (
                    info_containers[review_format["variety_index"]]
                    .find("div", {"class": "info"})
                    .span.findChildren()[0]
                    .contents[0]
                )
            except:
                raise ReviewFormatException("Unexpected variety format")
        else:
            variety = None

        if review_format["appellation_index"] is not None:
            appellation_info = (
                info_containers[review_format["appellation_index"]]
                .find("div", {"class": "info"})
                .span.findChildren()
            )
            try:
                if review_format["appellation_format"] == APPELLATION_FORMAT_0:
                    region_1 = None
                    region_2 = None
                    province = appellation_info[0].contents[0]
                    country = appellation_info[1].contents[0]
                elif review_format["appellation_format"] == APPELLATION_FORMAT_1:
                    region_1 = appellation_info[0].contents[0]
                    region_2 = None
                    province = appellation_info[1].contents[0]
                    country = appellation_info[2].contents[0]
                elif review_format["appellation_format"] == APPELLATION_FORMAT_2:
                    region_1 = appellation_info[0].contents[0]
                    region_2 = appellation_info[1].contents[0]
                    province = appellation_info[2].contents[0]
                    country = appellation_info[3].contents[0]
                else:
                    region_1 = None
                    region_2 = None
                    province = None
                    country = None
            except:
                raise ReviewFormatException("Unknown appellation format")
        else:
            region_1 = None
            region_2 = None
            province = None
            country = None

        if review_format["winery_index"] is not None:
            try:
                winery = (
                    info_containers[review_format["winery_index"]]
                    .find("div", {"class": "info"})
                    .span.span.findChildren()[0]
                    .contents[0]
                )
            except:
                raise ReviewFormatException("Unexpected winery format")
        else:
            winery = None

        review_data = {
            "points": points,
            "title": title,
            "description": description,
            "price": price,
            "designation": designation,
            "variety": variety,
            "region_1": region_1,
            "region_2": region_2,
            "province": province,
            "country": country,
            "winery": winery
        }

        return review_data

    def determine_review_format(self, review_soup):
        review_format = {}
        info_containers = review_soup.find("ul", {"class": "primary-info"}).find_all(
            "li", {"class": "row"}
        )

        review_info = []
        for container in info_containers:
            review_info.append(str(container.find("span").contents[0]).lower())

        try:
            review_format["price_index"] = review_info.index("price")
        except ValueError:
            review_format["price_index"] = None
        try:
            review_format["designation_index"] = review_info.index("designation")
        except ValueError:
            review_format["designation_index"] = None
        try:
            review_format["variety_index"] = review_info.index("variety")
        except ValueError:
            review_format["variety_index"] = None
        try:
            review_format["appellation_index"] = review_info.index("appellation")
        except ValueError:
            review_format["appellation_index"] = None
        try:
            review_format["winery_index"] = review_info.index("winery")
        except ValueError:
            review_format["winery_index"] = None

        # The appellation format changes based on where in the world the winery is located
        if review_format["appellation_index"] is not None:
            appellation_info = (
                info_containers[review_format["appellation_index"]]
                .find("div", {"class": "info"})
                .span.findChildren()
            )
            if len(appellation_info) == 2:
                review_format["appellation_format"] = APPELLATION_FORMAT_0
            elif len(appellation_info) == 3:
                review_format["appellation_format"] = APPELLATION_FORMAT_1
            elif len(appellation_info) == 4:
                review_format["appellation_format"] = APPELLATION_FORMAT_2
            else:
                review_format["appellation_format"] = UNKNOWN_FORMAT

        return review_format

    def clear_output_data(self):
        try:
            os.remove("{}.json".format(FILENAME))
        except FileNotFoundError:
            pass

class ReviewFormatException(Exception):
    """Exception when the format of a review page is not understood by the scraper"""

    def __init__(self, message):
        self.message = message
        super(Exception, self).__init__(message)


if __name__ == "__main__":
    
    # Total review results on their site are conflicting, hardcode as the max tested value for now
    winmag_scraper = Scraper(
        years_to_scrape=(2019, 2018, 2017),
        pages_to_scrape=(1000, 1254, 1, 1238, 1, 1154),
        clear_old_data=False
    )

    # Step 1: scrape review urls
    try:
        with open(URLFILENAME, "rb") as f:
            review_urls = pickle.load(f)
        # review_urls = winmag_scraper.scrape_site_for_urls(review_urls)
    except:
        review_urls = winmag_scraper.scrape_site_for_urls()
    
    # with open(URLFILENAME, "wb") as f:
    #     pickle.dump(review_urls, f)

    print("There were " + str(len(review_urls)) + " review urls found.")

    # Step 2: scrape data
    winmag_scraper.scrape_reviews(review_urls)
    print("Scrape Data finished...")

    # TODO: Only 85992 of 98700 have been scraped, but large possibility of duplicate values


  