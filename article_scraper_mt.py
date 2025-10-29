"""
Article scraper version implementing multithreading to speedup web scraping
"""

import time                     # delay between http requests
from tqdm import tqdm           # completition bars
import os                       # managing paths
import re                       # regular expressions for filtering
import utils                    # access to preprocess_text
import requests                 # http requests
from bs4 import BeautifulSoup   # parsing of reuests responses
import threading                # provides lock for multithreading
from concurrent.futures import ThreadPoolExecutor # implements multithreading

class SimpleRateLimiter:
    def __init__(self, per_sec: float = 2.0):
        self.per_sec = per_sec
        self._last_t = 0.0

    def wait(self):
        if self.per_sec <= 0:
            return
        now = time.time()
        min_dt = 1.0 / self.per_sec
        dt = now - self._last_t
        if dt < min_dt:
            time.sleep(min_dt - dt)
        self._last_t = time.time()


class ArticlesScraper:
    def __init__(self, stop_phrases, lang_check, requests_per_second=2.0, max_workers=5):
        self.stop_phrases = stop_phrases
        self.lang_check = lang_check
        self.rate_limiter = SimpleRateLimiter(per_sec=requests_per_second)
        self.max_workers = max_workers
        self.file_lock = threading.Lock()

    def _is_valid_paragraph(self, text:str):
        words = text.split()
        if len(words) <= 15: return False
        text_lower = text.lower()
        for phrase in self.stop_phrases:
            if phrase in text_lower: return False

        # Reject if it starts with characters common in lists or metadata
        if text.strip().startswith(('*', '#', '[')): return False

        # Reject if it looks like a decklist category
        if re.match(r'^\[\w+\]', text.strip()):
            return False

        # Reject if it looks like a timestamp
        if re.match(r'^\d{1,2}:\d{2}', text.strip()): return False

        # Reject if it contains a high ratio of capitalized words (likely a decklist)
        if len(words) > 1:
            capitalized_words = sum(1 for word in words[1:] if word and word[0].isupper())
            if capitalized_words / len(words) > 0.4:
                return False
            
            consecutive_caps = 0
            for word in words:
                # If we find 5 capitalized words in a row it's almost certainly a list
                if word and word[0].isupper():
                    consecutive_caps += 1
                    if consecutive_caps > 4:
                        return False 
                else:
                    consecutive_caps = 0
        
        # Reject common web/email patterns
        if re.search(r'https?://\S+|www\.\S+', text_lower): return False
        if re.search(r'\S+@\S+\.\S+', text_lower): return False
        return True
    

    def _get_all_article_links(self):
        sitemap_index_url = "https://edhrec.com/articles/sitemap_index.xml"
        print(f"Fetching sitemap pages from: {sitemap_index_url}")
        
        self.rate_limiter.wait()
        try:
            response = requests.get(sitemap_index_url, headers={'User-Agent': 'MTG-AI-Thesis-Scraper/1.0'})
            response.raise_for_status()
            index_soup = BeautifulSoup(response.content, 'lxml-xml')
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch the main sitemap index. Aborting. Error: {e}")
            return []

        all_sitemap_urls = [loc.text for loc in index_soup.find_all('loc')]
        article_sitemap_urls = [url for url in all_sitemap_urls if 'post-sitemap' in url]
        print(f"Found {len(article_sitemap_urls)} article-specific sitemaps to parse.")

        all_article_links = set()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=len(article_sitemap_urls), desc="Parsing Article Sitemaps") as pbar:
                futures = [
                    executor.submit(self._get_article_links, sitemap_url, all_article_links, pbar)
                    for sitemap_url in article_sitemap_urls
                ]
                for future in futures:
                    future.result()
    
        print(f"\nFound a total of {len(all_article_links)} unique article links from all sitemaps.")
        return list(all_article_links)

    def _get_article_links(self, sitemap_url:str, all_article_links, pbar):
        try:
            sitemap_response = requests.get(sitemap_url, headers={'User-Agent': 'MTG-AI-Thesis-Scraper/1.0'})
            sitemap_response.raise_for_status()
            sitemap_soup = BeautifulSoup(sitemap_response.content, 'lxml-xml')

            with self.file_lock:
                for loc_tag in sitemap_soup.select("url > loc"):
                    all_article_links.add(loc_tag.text)

        except requests.exceptions.RequestException as e:
            tqdm.write(f"Warning: Failed to parse sitemap {sitemap_url}. Skipping. Error: {e}")
        finally:
            pbar.update(1) 

        return
    
    def _scrape_and_save_article(self, article_url: str, output_path: str, pbar):
        """ Scraping of a single article and paragraph writing """
        paragraphs = []
        try:
            self.rate_limiter.wait()
            response = requests.get(article_url, headers={'User-Agent': 'MTG-AI-Thesis-Scraper/1.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            html_tag = soup.find('html')
            if not html_tag or 'en' not in html_tag.get('lang', ''):
                return

            content_div = soup.find('div', class_='wp-content')
            if not content_div:
                return

            for p_tag in content_div.find_all('p'):
                text = p_tag.get_text(" ", strip=True)
                if not text: continue
                text = re.sub(r'\[el\](.*?)\[/el\]', r'\1', text)
                
                # Check language, block all article if not written in english
                text_lower = text.lower().split(" ")
                for word in self.lang_check:
                    if word in text_lower:
                        return

                # Check validity of contents
                if self._is_valid_paragraph(text):
                    clean_text = utils.preprocess_text(text=text, mask_name=False)
                    clean_text = clean_text.replace("#", "")
                    if clean_text:
                        paragraphs.append(clean_text)

            # Saving results on file
            if paragraphs:
                with self.file_lock:
                    with open(output_path, 'a', encoding='utf-8') as f:
                        for p in set(paragraphs):
                            cleaned_p = utils.preprocess_text(text=p, mask_name=False).replace("#", "")
                            if cleaned_p:
                                f.write(cleaned_p + '\n')

        except Exception as e:
            tqdm.write(f"Warning: Failed to process URL {article_url}. Skipping. Error: {e}")
        finally:
            pbar.update(1)

    def scrape_articles_into_paragraphs(self, output_path:str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Removing already existing paragraphs if needed
        if os.path.exists(output_path):
            os.remove(output_path)

        all_article_links = self._get_all_article_links()
        if not all_article_links:
            print("No article links to process. Exiting.")
            return

        print(f"Scraping {len(all_article_links)} articles using {self.max_workers} threads...")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(total=len(all_article_links), desc="Scraping Articles") as pbar:
                futures = [
                    executor.submit(self._scrape_and_save_article, link, output_path, pbar)
                    for link in all_article_links
                ]
                for future in futures:
                    future.result() 
        
        print("\nScraping complete.")


class Params:
    requests_per_second = 3.0
    max_workers = 32 
    stop_phrases = [
        "check out the data", "in the comments", "follow us", "patreon", "calendar", "i'll see you all", "we are all", "this morning",
        "thanks for reading", "your thoughts", "join our discord", "sound off below", "subscribe", "what will you do",
        "you can listen to cco podcast", "huge thank you to our sponsors", "decklists are provided below", "proxy", "proxies",
        "tell us", "hit me up", "use the code", "use our code", "stay tuned", "side events", "check out", "what commander do you play",
        "suggestions for any future deck ideas", "send me", "send us" "po box", "last week", "last month", "social media", "thanks so much",
        "sponsored by", "affiliate link", "promo code", "fan content", "not approved/endorsed by wizards", "$", "your order",
        "wizards of the coast", "trademarks of wizards", "all rights reserved", "property of wizards", "join us", "let you know",
        "timestamps:", "chapters", "directed by:", "executive producer", "edited by:", "credits", "this year", "new year", "last year",
        "this video", "this episode", "this week", "show notes", "let me know", "let us know", "commentary", "MTGOTraders",
        "intro & présentation", "debrief", "title sequence by", "music", "vfx artist", "next time", "for the life of", "my name is",
        "decklist provided by", "this series is brought to you by", "let's find out", "reddit", "twitter", "what do you think",
        "this channel", "greetings", "hello everyone", "hello, everyone", "welcome back", "welcome to", "lets look at", "who will take victory",
        "post new videos", "every week", "taken from the yt-audio-library", "huge thanks to", "podcast", "shout out to","shoutout to", "shout-out to",
        "i do not own any of the art", "content spike", "welcome to the command zone", "game knights", "this article", "this post",
        "commander's quarters", "the spike feeders", "jolt mtg", "mtg muddstah", "commander cookout", "commander's herald", "our channel",
        "nitpicking nerds", "jumbo commander", "the command sphere", "i hate your deck", "thank you for", "game night", "hey everyone", "hi everyone"
    ]
    lang_check = [
        "au", "aux", "al", "como", "con", "dans", "de", "del", "dem", "des", "della", "di", "das", "der",
        "e", "el", "en", "est", "et", "für", "gli", "il", "im", "ist", "la", "le", "les", "lo", "los",
        "más", "mit", "ne", "nicht", "pas", "per", "pour", "para", "que", "qui", "si", "se",
        "sind", "sono", "un", "una", "une", "y", "zu", "auf", "ce", "ces", "ein", "einige", "ist"
    ]


if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "data", "scraped_articles_mt.txt")

    articles_scraper = ArticlesScraper(
        stop_phrases=Params.stop_phrases, 
        lang_check=Params.lang_check,
        requests_per_second=Params.requests_per_second,
        max_workers=Params.max_workers
    )
    
    start_time = time.time()
    
    articles_scraper.scrape_articles_into_paragraphs(output_file)
    
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    print(f"\Start time: {start_time} seconds")
    print(f"\End time:   {end_time} seconds")
    print(f"\nScript execution time: {execution_time:.4f} seconds")