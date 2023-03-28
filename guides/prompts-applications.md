# 提示应用

在本指南中，我们将介绍一些高级和有趣的方式，利用提示工程来执行更有用和更高级的任务。

**请注意，该部分正在大力开发中。**
主题：
- [生成数据](#生成数据)
- [程序辅助语言模型](#pal程序辅助语言模型)
- [Python笔记本](#python笔记本)


---
## 生成数据
LLMs有强大的文本生成能力。使用有效的提示策略可以引导模型产生更好、更一致和更准确的回复。LLMs还可以特别有用于生成有用于运行各种实验数据。例如，我们可以使用它生成快速样本以用于情感分类器，就像这样：

*提示:*
```
生成10个情感分析例子。将示例归类为积极或消极。生成2个消极的示例和8个积极的示例。使用以下格式进行示例：
问: <句子>
答: <情感>
```
*输出:*
```
问: 我刚刚得到了最好的消息！
答: 积极

问: 我们在工作中得到了加薪！
答: 积极

问: 我为今天所取得的成就感到自豪。
答: 积极

问: 我有史以来过得最好的一天！
答: 积极

问: 我真的很期待周末。
答: 积极

问: 我刚收到了最好的礼物！
答: 积极

问: 我现在非常开心。
答: 积极

问: 我非常幸运能有这样一个神奇的家人。
答: 积极

问: 外面的天气真的很阴暗。
答: 消极

问: 我刚得到了一些可怕的消息。
答: 消极
```

这非常有用。我们在指南的另一部分中使用这个示例进行不同的测试。

---

## PAL（程序辅助语言模型）[Gao等，(2022)](https://arxiv.org/abs/2211.10435)提出了一种使用LLMs阅读自然语言问题并生成程序作为中间推理步骤的方法。被称为“程序辅助语言模型（PAL）”，与思维链提示不同的是，它不是使用自由格式文本来获得解决方案，而是将解决步骤卸载到编程运行时，如Python解释器。

![](../img/pal.png)

让我们看一个使用LangChain和OpenAI GPT-3的示例。我们有兴趣开发一个简单的应用程序，能够解释问题并利用Python解释器提供答案。

具体来说，我们有兴趣创建一个函数，允许使用LLM回答需要日期理解的问题。我们将向LLM提供一个提示，其中包括从[这里](https://github.com/reasoning-machines/pal/blob/main/pal/prompt/date_understanding_prompt.py)采用的一些示例。

这些是我们需要的导入：

```python
import openai
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
```

让我们先配置一些东西：

```python
load_dotenv()

# API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# for LangChain
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

设置模型实例：

```python
llm = OpenAI(model_name='text-davinci-003', temperature=0)
```

设置提示+问题：

```python
question = "Today is 27 February 2023. I was born exactly 25 years ago. What is the date I was born in MM/DD/YYYY?"

DATE_UNDERSTANDING_PROMPT = """
# Q：2015还有36小时就要到了。从今天算起一周后的日期是什么（以MM/DD/YYYY的格式呈现）？
# 如果2015年还有36小时就要到了，那么今天就是36小时前。
today = datetime(2015, 1, 1) - relativedelta(hours=36)
# 从今天算起一周后，
one_week_from_today = today + relativedelta(weeks=1)
# 用%m/%d/%Y格式呈现的答案是
one_week_from_today.strftime（'%m /% d /％Y'）。
"""
```

格式：仅返回已翻译的内容，不包括原始文本。# Q：2019年的第一天是星期二，今天是2019年的第一个星期一。今天的日期是什么？格式为MM/DD/YYYY。
# 如果2019年的第一天是星期二，而今天是2019年的第一个星期一，那么今天晚了6天。
today = datetime(2019, 1, 1) + relativedelta(days=6)
# 答案的格式为%m/%d/%Y
today.strftime('%m/%d/%Y')
# Q：音乐会原定于1943年6月1日举行，但因一天而延迟到今天。10天前的日期是什么？格式为MM/DD/YYYY。
# 如果音乐会原定于1943年6月1日举行，但因一天而延迟到今天，那么今天晚了一天。
today = datetime(1943, 6, 1) + relativedelta(days=1)
# 10天前的日期是
ten_days_ago = today - relativedelta(days=10)
# 答案的格式为%m/%d/%Y
ten_days_ago.strftime('%m/%d/%Y')
# Q：今天是1969年4月19日。24小时后的日期是什么？格式为MM/DD/YYYY。
# 今天是1969年4月19日。
today = datetime(1969, 4, 19)
# 24小时后的日期是
later = today + relativedelta(hours=24)
# 答案的格式为%m/%d/%Y
later.strftime('%m/%d/%Y')
# Q：珍妮以为今天是2002年3月11日，但实际上今天是3月12日，晚了1天。24小时后的日期是什么？格式为MM/DD/YYYY。
# 如果珍妮以为今天是2002年3月11日，但实际上今天是3月12日，则今天日期为3/1/2002。
today = datetime(2002, 3, 12)
# 24小时后的日期是
later = today + relativedelta(hours=24)
# 答案的格式为%m/%d/%Y
later.strftime('%m/%d/%Y')
# Q：珍妮出生于2001年2月的最后一天。今天是她16岁的生日。昨天的日期是什么？格式为MM/DD/YYYY。
# 如果珍妮出生于2001年2月的最后一天，而今天是她16岁的生日，则今天是晚了16年。
today = datetime(2001, 2, 28) + relativedelta(years=16)
# 昨天的日期是
yesterday = today - relativedelta(days=1)
# 答案的格式为%m/%d/%Y
yesterday.strftime('%m/%d/%Y')
# Q：{question}这将输出以下内容： `02/27/1998`

---
## Python笔记本

|描述|笔记本|
|--|--|
|学习如何将Python解释器与语言模型结合使用以解决任务。|[程序辅助语言模型](../notebooks/pe-pal.ipynb)|

---

更多示例即将推出！

[上一节（高级提示）](./prompts-advanced-usage.md)

[下一节（ChatGPT）](./prompts-chatgpt.md)