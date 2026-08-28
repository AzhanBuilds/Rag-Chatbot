from flask import Flask, render_template, request
import markdown
from rag_pipeline import answer_question

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form.get("question", "").strip()

        if question:
            answer = answer_question(question)

            # Convert Markdown to HTML
            answer = markdown.markdown(answer)            

    return render_template(
        "chat.html",
        question=question,
        answer=answer
    )


if __name__ == "__main__":
    app.run(debug=False)