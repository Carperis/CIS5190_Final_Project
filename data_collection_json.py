import requests
import json
import csv
import requests
import glob
import os


def scrape_fox_json_response(url, callback):
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.text
        json_data = json_data[len(callback) + 1 : -1]
        data = json.loads(json_data)
        return data
    else:
        print(f"Failed to retrieve data: {response.status_code, url}")
        return None


def scrape_nbc_json_response(url, callback, cookie=None):
    headers = {
        "Cookie": cookie,
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Dnt": "1",
        "Referer": "https://www.nbcnews.com/",
        "Sec-Fetch-Dest": "script",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "cross-site",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1.1 Safari/605.1.15",
    }
    if cookie is None:
        response = requests.get(url)
    else:
        response = requests.get(url, headers=headers)
    if response.status_code == 200:
        json_data = response.text
        json_data = json_data[len(callback)+1 : -2].replace("\n", "").strip()
        data = json.loads(json_data)
        return data
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

def process_fox_json(data):
    titles = set()
    title_count = 0
    item_count = len(data["data"])
    for item in data["data"]:
        if item['type'] == 'article':
            title = item["attributes"]['title']
            titles.add(title)
            title_count += 1
    return titles, title_count, item_count

def process_nbc_json(data):
    titles = set()
    title_count = 0
    item_count = len(data["results"])
    for item in data["results"]:
        try: 
            if item["richSnippet"]["metatags"]["ogType"] == "article":
                title = item["richSnippet"]["metatags"]["ogTitle"]
                titles.add(title)
                title_count += 1
        except:
            pass
    return titles, title_count, item_count

def save_data(output, titles, news):
    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "news"])
        for title in titles:
            writer.writerow([title, news])

def concat_data(fox_file, nbc_file, output):
    with open(fox_file, "r") as f:
        fox_data = f.readlines()
    with open(nbc_file, "r") as f:
        nbc_data = f.readlines()
    fox_len = len(fox_data) - 1
    nbc_len = len(nbc_data) - 1
    fox_check = set()
    nbc_check = set()
    with open(output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["title", "news"])
        for line in fox_data[1:]:
            title = line.strip().split(",f")[0]
            news = "f" + line.strip().split(",f")[1]
            if title in fox_check:
                raise(f"Fox data has duplicates: {title}")
            fox_check.add(title)
            writer.writerow([title, news])
        for line in nbc_data[1:]:
            title = line.strip().split(",n")[0]
            news = "n" + line.strip().split(",n")[1]
            if title in nbc_check:
                raise(f"NBC data has duplicates: {title}")
            nbc_check.add(title)
            writer.writerow([title, news])
    assert len(fox_check) == fox_len, "Fox data has duplicates"
    assert len(nbc_check) == nbc_len, "NBC data has duplicates"

def generate_search_query():
    search = []
    for year in range(2019, 2023):
        for month in range(1, 13):
            for day in range(1, 32):
                search.append(f"{year}-{month:02d}-{day:02d}")
    return search

def save_nbc_temp_txt(titles, query):
    for file in glob.glob("nbc_temp*.txt"):
        os.remove(file)
    filename = f"nbc_temp_{query}.txt"
    with open (filename, "w") as f:
        for title in titles:
            f.write(title + "\n")

if __name__ == "__main__":
    target_num = 10000
    fox_output = f"./data/fox_data{target_num}.csv"
    nbc_output = f"./data/nbc_data{target_num}.csv"
    output = f"./data/data{target_num*2}.csv"
    search = generate_search_query()

    fox_titles = set()
    index = 0
    while len(fox_titles) < target_num:
        if index >= len(search):
            break
        query = search[index]
        prev_len = -1
        start_index = 1
        query_count = 1
        while len(fox_titles) > prev_len:
            prev_len = len(fox_titles)
            print(f"Scraping Fox News: {len(fox_titles)} / {target_num} ({query}, {start_index}, {query_count})"+""*10, end="\r")
            fox_url = f"https://moxie.foxnews.com/search/web?q={query}&start={start_index}&callback=__jp{query_count}"
            data = scrape_fox_json_response(fox_url, f"__jp{query_count}")
            titles, title_count, item_count = process_fox_json(data)
            fox_titles.update(titles)
            start_index += item_count
            query_count += 1
        index += 1
    save_data(fox_output, fox_titles, "fox")
    print("Scraping Fox News: Done")

    nbc_titles = set()
    index = 0
    if len(glob.glob("nbc_temp*.txt")) > 0:
        filename = glob.glob("nbc_temp*.txt")[0]
        with open(filename, "r") as f:
            index = search.index(filename.split("_")[2].split(".")[0])
            for line in f.readlines():
                nbc_titles.add(line.strip())
    while len(nbc_titles) < target_num:
        if index >= len(search):
            break
        query = search[index]
        prev_len = -1
        start_index = 0
        while len(nbc_titles) > prev_len and start_index < 120:
            prev_len = len(nbc_titles)
            print(f"Scraping NBC News: {len(nbc_titles)} / {target_num} ({query}, {start_index})"+""*10, end="\r")
            cookie = "NID=519=bLdC_THZ0s3d9BFirBmSR0wmeecLmHsdlB9i4GEt34fBYHAb9nljN_YbSOjuGOObQGVa7834zgJUIkWGZxrcoL1sXXLAZ9Ddsx8iiaaQF9s213nkN34RPgQ8DAIeO1CSJOBG2XjOJVX4lXPmLJFIbnjtHgapbRPPUD6TFCVnyKXtKIYkpVpaJfufswS3q8A"
            nbc_url = "https://cse.google.com/cse/element/v1?" + \
                "rsz=filtered_cse&" + \
                "num=20&hl=en&" + \
                "source=gcsc&" + \
                f"start={start_index}&" + \
                "cselibv=8fa85d58e016b414&" + \
                "cx=003487489644438443209%3Arbq9uxjpv_m&" + \
                f"q={query}&" + \
                "safe=off&" + \
                "cse_tok=AB-tC_5ViPkxa_4C3VLmPcaYqWI6%3A1732503509961&" + \
                "sort=&" + \
                "exp=cc%2Capo&" + \
                "fexp=72801196%2C72801194%2C72801195&" + \
                "callback=google.search.cse.api2248&" + \
                f"rurl=https%3A%2F%2Fwww.nbcnews.com%2Fsearch%2F%3Fq%3D{query}&" + \
                "nocache=1732503510953"
            data = scrape_nbc_json_response(nbc_url, "/*O_o*/\ngoogle.search.cse.api2248", cookie=cookie)
            if data is not None and "error" in data and data["error"]["code"] == 429:
                save_nbc_temp_txt(nbc_titles, query)
                raise(f"Rate Limited: ({query}, {start_index})")
            if data is None or "results" not in data:
                save_nbc_temp_txt(nbc_titles, query)
                break
            titles, title_count, item_count = process_nbc_json(data)
            nbc_titles.update(titles)
            start_index += item_count
        index += 1
    for file in glob.glob("nbc_temp*.txt"):
        os.remove(file)
    save_data(nbc_output, nbc_titles, "nbc")
    print("Scraping NBC News: Done")

    concat_data(fox_output, nbc_output, output)
