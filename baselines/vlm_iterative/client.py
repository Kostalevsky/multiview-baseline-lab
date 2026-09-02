import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class VLMClient:
    def select_next_view(
        self,
        candidate_collage,
        selected_collage,
        prompt,
        remaining_indices,
    ):
        raise NotImplementedError


class MockVLMClient(VLMClient):
    def select_next_view(
        self,
        candidate_collage,
        selected_collage,
        prompt,
        remaining_indices,
    ):
        if not remaining_indices:
            raise RuntimeError("No remaining candidate views.")

        selected_index = remaining_indices[0]

        return {
            "selected_index": selected_index,
            "reason": ("Mock selection for pipeline testing."),
        }


class QwenVLMClient(VLMClient):
    def __init__(
        self,
        model=None,
        api_key=None,
        base_url=None,
    ):
        self.model = model or os.environ.get("QWEN_VLM_MODEL")

        self.api_key = api_key or os.environ.get("QWEN_API_KEY")

        self.base_url = base_url or os.environ.get("QWEN_BASE_URL")

        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY is not set.")

        if not self.base_url:
            raise RuntimeError("QWEN_BASE_URL is not set.")

        if not self.model:
            raise RuntimeError("QWEN_VLM_MODEL is not set.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def encode_image(
        self,
        image_path,
    ):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: " f"{image_path}")

        with image_path.open("rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")

        suffix = image_path.suffix.lower()

        if suffix == ".png":
            mime_type = "image/png"
        elif suffix in (
            ".jpg",
            ".jpeg",
        ):
            mime_type = "image/jpeg"
        else:
            raise RuntimeError("Unsupported image format: " f"{suffix}")

        return f"data:{mime_type};" f"base64,{encoded}"

    def build_content(
        self,
        candidate_collage,
        selected_collage,
        prompt,
    ):
        content = []

        content.append(
            {
                "type": "text",
                "text": ("Candidate views:"),
            }
        )

        candidate_image = self.encode_image(candidate_collage)

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": candidate_image,
                },
            }
        )

        if selected_collage is not None:
            content.append(
                {
                    "type": "text",
                    "text": ("Views already selected:"),
                }
            )

            selected_image = self.encode_image(selected_collage)

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": selected_image,
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        return content

    def parse_response(
        self,
        response_text,
        remaining_indices,
    ):
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Qwen returned invalid JSON:\n" f"{response_text}"
            ) from error

        selected_index = result.get("selected_index")

        reason = result.get(
            "reason",
            "",
        )

        if not isinstance(
            selected_index,
            int,
        ):
            raise RuntimeError(
                "Qwen response does not " "contain an integer " "'selected_index'."
            )

        if selected_index not in (remaining_indices):
            raise RuntimeError(
                "Qwen selected invalid " f"candidate index: " f"{selected_index}"
            )

        return {
            "selected_index": selected_index,
            "reason": result.get(
                "reason",
                "",
            ),
            "covered_geometry": result.get(
                "covered_geometry",
                [],
            ),
            "missing_geometry": result.get(
                "missing_geometry",
                [],
            ),
        }

    def select_next_view(
        self,
        candidate_collage,
        selected_collage,
        prompt,
        remaining_indices,
    ):
        if not remaining_indices:
            raise RuntimeError("No remaining candidate views.")

        content = self.build_content(
            candidate_collage=(candidate_collage),
            selected_collage=(selected_collage),
            prompt=prompt,
        )

        print("[VLM Iterative] " f"Calling Qwen model: " f"{self.model}")

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are an expert "
                                "in 3D geometry, "
                                "CAD objects, and "
                                "multi-view analysis. "
                                "You must select "
                                "camera viewpoints "
                                "based only on the "
                                "provided images."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            response_format={"type": ("json_object")},
            temperature=0.0,
        )

        response_text = completion.choices[0].message.content

        if not response_text:
            raise RuntimeError("Qwen returned an " "empty response.")

        print("[VLM Iterative] " "Qwen response:")

        print(response_text)

        return self.parse_response(
            response_text=(response_text),
            remaining_indices=(remaining_indices),
        )
    
    def review_final_set(
        self,
        selected_collage,
        candidate_collage,
        prompt,
        selected_indices,
        remaining_indices,
    ):
        content = []

        content.append(
            {
                "type": "text",
                "text": "Current selected views:",
            }
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": self.encode_image(
                        selected_collage
                    ),
                },
            }
        )

        content.append(
            {
                "type": "text",
                "text": "Remaining candidate views:",
            }
        )

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": self.encode_image(
                        candidate_collage
                    ),
                },
            }
        )

        content.append(
            {
                "type": "text",
                "text": prompt,
            }
        )

        completion = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are an expert in "
                                    "3D geometry and "
                                    "multi-view analysis."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": content,
                    },
                ],
                response_format={
                    "type": "json_object"
                },
                temperature=0.0,
            )
        )

        response_text = (
            completion
            .choices[0]
            .message
            .content
        )

        result = json.loads(
            response_text
        )

        replace = result.get(
            "replace",
            False,
        )

        if not replace:
            return {
                "replace": False,
                "reason": result.get(
                    "reason",
                    "",
                ),
            }

        remove_index = result.get(
            "remove_index"
        )

        add_index = result.get(
            "add_index"
        )

        if remove_index not in (
            selected_indices
        ):
            raise RuntimeError(
                "Qwen returned invalid "
                f"remove_index: "
                f"{remove_index}"
            )

        if add_index not in (
            remaining_indices
        ):
            raise RuntimeError(
                "Qwen returned invalid "
                f"add_index: "
                f"{add_index}"
            )

        return {
            "replace": True,
            "remove_index": remove_index,
            "add_index": add_index,
            "reason": result.get(
                "reason",
                "",
            ),
        }