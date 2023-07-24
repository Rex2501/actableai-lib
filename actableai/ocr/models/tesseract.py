from typing import Iterable

import ray
from PIL.Image import Image

from actableai.ocr.models.base import BaseOCRModel, Model
from actableai.ocr.serve.tesseract import TesseractServe, TesseractModelType
from actableai.parameters.options import OptionsParameter
from actableai.parameters.parameters import Parameters
from actableai.utils.language import get_language_display_name


class Tesseract(BaseOCRModel):
    @staticmethod
    def get_parameters() -> Parameters:
        serve_handle = TesseractServe.get_handle()
        available_languages = ray.get(
            serve_handle.options(method_name="get_available_languages").remote(
                model_type=TesseractModelType.normal
            )
        )

        parameters = [
            OptionsParameter[str](
                name="languages",
                display_name="Languages",
                descritption="Language models to use. Note: Selecting multiple languages is slowing down the inference.",
                default="eng",
                is_multi=True,
                options={
                    lang_code: {
                        "display_name": get_language_display_name(lang_code),
                        "value": lang_code,
                    }
                    for lang_code in available_languages
                },
            ),
            OptionsParameter[str](
                name="model_type",
                display_name="Model Type",
                description="Type of model to choose, 'fast' is the fastest and 'best' is the slowest. Note: A slow model type will result in better performances.",
                default="best",
                is_multi=False,
                options={
                    "fast": {
                        "display_name": "Fast",
                        "value": "fast",
                    },
                    "normal": {
                        "display_name": "Normal",
                        "value": "normal",
                    },
                    "best": {
                        "display_name": "Best",
                        "value": "best",
                    },
                },
            ),
            OptionsParameter[int](
                name="page_segmentation_mode",
                display_name="Page Segmentation Mode",
                desription="Advanced feature, choose the mode used for the page segmentation",
                default="3",
                is_multi=False,
                options={
                    "1": {
                        "display_name": "Orientation and script detection (OSD) only.",
                        "value": 1,
                    },
                    "3": {
                        "display_name": "Fully automatic page segmentation, but no OSD.",
                        "value": 3,
                    },
                    "4": {
                        "display_name": "Assume a single column of text of variable sizes.",
                        "value": 4,
                    },
                    "5": {
                        "display_name": "Assume a single uniform block of vertically aligned text.",
                        "value": 5,
                    },
                    "6": {
                        "display_name": "Assume a single uniform block of text.",
                        "value": 6,
                    },
                    "7": {
                        "display_name": "Treat the image as a single text line.",
                        "value": 7,
                    },
                    "8": {
                        "display_name": "Treat the image as a single word.",
                        "value": 8,
                    },
                    "9": {
                        "display_name": "Treat the image as a single word in a circle.",
                        "value": 9,
                    },
                    "10": {
                        "display_name": "Treat the image as a single character.",
                        "value": 10,
                    },
                    "11": {
                        "display_name": "Sparse text. Find as much text as possible in no particular order.",
                        "value": 11,
                    },
                    "12": {
                        "display_name": "Sparse text with OSD.",
                        "value": 12,
                    },
                    "13": {
                        "display_name": "Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.",
                        "value": 13,
                    },
                },
            ),
        ]

        return Parameters(
            name=Model.tesseract,
            display_name="Tesseract",
            parameters=parameters,
        )

    def _transform(self, data: Iterable[Image]) -> Iterable[str]:
        model_type = TesseractModelType(self.parameters["model_type"])
        language_list = self.parameters["languages"]
        page_segmentation_mode = int(self.parameters["page_segmentation_mode"])

        serve_handle = TesseractServe.get_handle()

        # TODO look into making this parallelized, but as long as we only have one actor
        #   this will not make a difference as tesseract can only handle one image at a
        #   time
        return [
            ray.get(
                serve_handle.options(method_name="transform").remote(
                    image=image,
                    model_type=model_type,
                    lang="+".join(language_list)
                    if isinstance(language_list, list)
                    else language_list,
                    page_segmentation_mode=page_segmentation_mode,
                )
            )
            for image in data
        ]
