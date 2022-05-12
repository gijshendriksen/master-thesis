from lxml import etree

from feature_extraction.base import BaseExtractor


class TextExtractor(BaseExtractor):
    output_format = '{split}/{vertical}/{website}-{max_length}.csv'

    def feature_representation(self, elem: etree.Element) -> str:
        return self.text_representation(elem)