<div id="top"/>
<!-- PROJECT SHIELDS -->

[![Actions Status](https://github.com/Actable-AI/actableai-lib/workflows/UnitTest/badge.svg)](https://github.com/Actable-AI/actableai-lib/actions)
[![Actions Status](https://github.com/Actable-AI/actableai-lib/workflows/Release%20API%20Docs/badge.svg)](https://github.com/Actable-AI/actableai-lib/actions)

##

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://actable.ai" target="_blank">
    <img src="https://docs.actable.ai/_images/logo.png" alt="Logo">

  <p align="center">
    Advanced Analytics and Data Science made easy
    <br />
    <a href="https://lib.actable.ai" target="_blank"><strong>Explore the docs</strong></a>
    <br />
    <br />
    <a href="https://app.actable.ai/superset/explore/?form_data=%7B%22datasource%22%3A%2228506__table%22%2C%22viz_type%22%3A%22plotly_correlation%22%2C%22slice_id%22%3A623%2C%22url_params%22%3A%7B%7D%2C%22time_range_endpoints%22%3A%5B%22inclusive%22%2C%22exclusive%22%5D%2C%22adhoc_filters%22%3A%5B%5D%2C%22correlation_target%22%3A%22two_year_recid%22%2C%22columns_name%22%3A%5B%22compas_screening_date%22%2C%22sex%22%2C%22dob%22%2C%22age%22%2C%22age_cat%22%2C%22race%22%2C%22juv_fel_count%22%2C%22juv_misd_count%22%2C%22juv_other_count%22%2C%22priors_count%22%2C%22days_b_screening_arrest%22%2C%22c_jail_in%22%2C%22c_jail_out%22%2C%22c_case_number%22%2C%22c_offense_date%22%2C%22c_arrest_date%22%2C%22c_days_from_compas%22%2C%22c_charge_degree%22%2C%22c_charge_desc%22%2C%22r_days_from_arrest%22%2C%22r_offense_date%22%2C%22r_charge_desc%22%2C%22r_jail_in%22%2C%22r_jail_out%22%2C%22violent_recid%22%2C%22is_violent_recid%22%2C%22vr_case_number%22%2C%22vr_charge_degree%22%2C%22vr_offense_date%22%2C%22vr_charge_desc%22%2C%22type_of_assessment%22%2C%22screening_date%22%2C%22in_custody%22%2C%22out_custody%22%2C%22start%22%2C%22end%22%2C%22event%22%5D%2C%22correlation_control%22%3A%5B%5D%2C%22number_factors%22%3A20%2C%22show_bar_value%22%3Afalse%2C%22taskId%22%3A%22f1e3c2e7-b093-460f-9801-d69c98dbcd54%22%2C%22sql%22%3Anull%2C%22databaseName%22%3A%22actableai%22%7D" target="_blank">View Demo</a>
    ·
    <a href="https://github.com/Actable-AI/actableai-lib/issues" target="_blank">Report Bug</a>
    ·
    <a href="https://github.com/Actable-AI/actableai-lib/issues" target="_blank">Request Feature</a>
  </p>
</div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache v2.0 License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

**actableai-lib** is a Python module for automated Machine Learning and Analytics. It offers a wide range of *tasks* that automatically infer statistics or results based on your data.

This repository is a library used for our main app [app.actable.ai](https://app.actable.ai) where you can run every analytics and inferences without coding.

This project is started and maintained by [Actable AI](https://actable.ai).

### Built With

* [Autogluon](https://auto.gluon.ai/)
* [Ray](https://docs.ray.io/)
* [Sklearn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org//)

### Environment

* [X] [Ubuntu>18.04](https://wiki.ubuntu.com/Releases)
* [X] MacOS

### Prerequisites

* You need to have [python 3.8](https://www.python.org/downloads/release/python-380) installed with [pip](https://pip.pypa.io/en/stable/)
* You need to have [R](https://www.r-project.org/) installed
* You need to have multi_rake installed in your python env :
One way to do so is `pip install git+https://github.com/Actable-AI/multi_rake@master#egg=multi_rake`
* You need to have [tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
* If you are using the gpu version, you need CUDA 11.0 installed for mxnet

### Installation

* Option 1 - Install from PyPI :

```sh
pip install actableai-lib[gpu]
```

* Option 2 - Install from source :

```sh
git clone git@github.com:Actable-AI/actableai-lib.git --recursive
cd actableai-lib
pip install .[gpu]
```

To contribute to the repo, when installing from source, run `pip install -e .[gpu,dev]` instead of `pip install .[gpu]` to enable pip's developer mode.

You can also replace `[gpu]` with `[cpu]` to use the cpu only version.

<!-- USAGE EXAMPLES -->
## Usage

* Running a Classification :

```python
import pandas as pd
from actableai.tasks.classification import AAIClassificationTask

df = pd.read_csv("path/to/dataframe.csv")
result = AAIClassificationTask().run(
  df,
  target='target_column'
)
```

* Running a Correlation Analysis :

```python
import pandas as pd
from actableai.tasks.correlation import AAICorrelationTask

df = pd.read_csv("path/to/dataframe.csv")
result = AAICorrelationTask().run(
  df,
  target='target_column'
)
```

_For detailed examples and common use cases, please refer to the [Documentation](https://lib.actable.ai)_

<!-- ROADMAP -->
## Roadmap

* [ ] Add Changelog
* [X] Add LICENSE
* [X] Add PyPI installer
* [ ] Survival Analysis
* [ ] Multi Armed Bandit

See the [open issues](https://github.com/Actable-AI/actableai-lib/issues) for a full list of proposed features (and known issues).

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

To contribute to the code base, after cloning :

1. Install our git hooks by running locally :

```sh
./scripts/setup_hooks.sh
```

2. Create your Feature/BugFix Branch: `git checkout -b feature/AmazingFeature`
3. Commit your Changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the Branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

Your pull request will be reviewed asap by an Actable AI maintainer.

<!-- LICENSE -->
## License

Distributed under the License Apache v2.0. See `LICENSE` file for more information.

<!-- CONTACT -->
## Contact

Trung Huynh - [@Linkedin](https://www.linkedin.com/in/trunghlt/) - trung@actable.ai

Project Link: [https://github.com/Actable-AI/actableai-lib](https://github.com/Actable-AI/actableai-lib)

## Official Maintainers

For any question about the ML Library feel free to send us a message :

* trung@actable.ai
* christiang@actable.ai


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/Actable-AI/actableai-lib.svg?style=for-the-badge
[contributors-url]: https://github.com/Actable-AI/actableai-lib/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Actable-AI/actableai-lib.svg?style=for-the-badge
[forks-url]: https://github.com/Actable-AI/actableai-lib/network/members
[stars-shield]: https://img.shields.io/github/stars/Actable-AI/actableai-lib.svg?style=for-the-badge
[stars-url]: https://github.com/Actable-AI/actableai-lib/stargazers
[issues-shield]: https://img.shields.io/github/issues/Actable-AI/actableai-lib.svg?style=for-the-badge
[issues-url]: https://github.com/Actable-AI/actableai-lib/issues
[license-shield]: https://img.shields.io/github/license/Actable-AI/actableai-lib.svg?style=for-the-badge
[license-url]: https://github.com/Actable-AI/actableai-lib/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=256
[linkedin-url]: https://www.linkedin.com/company/actable-ai
