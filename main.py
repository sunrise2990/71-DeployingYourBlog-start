# main.py
from datetime import datetime
from datetime import date
from flask import Flask, abort, render_template, redirect, url_for, flash, request
from flask_bootstrap import Bootstrap5
from flask_migrate import Migrate
from flask_ckeditor import CKEditor
from flask_gravatar import Gravatar
from flask_login import UserMixin, login_user, LoginManager, current_user, logout_user
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, func, JSON, DateTime
from sqlalchemy.sql import func
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from forms import CreatePostForm, RegisterForm, LoginForm, CommentForm
import os
import logging
from dotenv import load_dotenv
from routes import projects_bp
from routes import scenarios_bp



# 🔧 Initialize logging + environment
logging.basicConfig(level=logging.DEBUG)
load_dotenv()

# 🔁 Internal imports (deferred to avoid circular dependency)
from models import db
from models.stock import etl
from models.stock.etl import load_stock_data
from models.stock.stock_routes import bp_stock

# ✅ App configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DB_URI")
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")
db.init_app(app)
migrate = Migrate(app, db)
app.register_blueprint(projects_bp)
app.register_blueprint(scenarios_bp)

# ✅ Extensions
ckeditor = CKEditor(app)
Bootstrap5(app)
app.register_blueprint(bp_stock)

# ✅ Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return db.get_or_404(User, user_id)

# ✅ Gravatar for user profile images
gravatar = Gravatar(app,
                    size=100,
                    rating='g',
                    default='retro',
                    force_default=False,
                    force_lower=False,
                    use_ssl=False,
                    base_url=None)

# ✅ Base class for models
class Base(DeclarativeBase):
    pass

# ✅ BlogPost table
class BlogPost(db.Model):
    __tablename__ = "blog_posts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    author_id: Mapped[int] = mapped_column(Integer, db.ForeignKey("users.id"))
    author = relationship("User", back_populates="posts")
    title: Mapped[str] = mapped_column(String(250), unique=True, nullable=False)
    subtitle: Mapped[str] = mapped_column(String(250), nullable=False)
    category = db.Column(db.String(100), nullable=True)
    date: Mapped[str] = mapped_column(String(250), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    img_url: Mapped[str] = mapped_column(String(250), nullable=False)
    comments = relationship("Comment", back_populates="parent_post")

# ✅ User table
class User(UserMixin, db.Model):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(100), unique=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str] = mapped_column(String(100))
    posts = relationship("BlogPost", back_populates="author")
    comments = relationship("Comment", back_populates="comment_author")

# ✅ Comment table
class Comment(db.Model):
    __tablename__ = "comments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    author_id: Mapped[int] = mapped_column(Integer, db.ForeignKey("users.id"))
    comment_author = relationship("User", back_populates="comments")
    post_id: Mapped[str] = mapped_column(Integer, db.ForeignKey("blog_posts.id"))
    parent_post = relationship("BlogPost", back_populates="comments")


# ✅ Create tables
with app.app_context():
    db.create_all()

# ✅ Admin-only decorator
def admin_only(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.id != 1:
            return abort(403)
        return f(*args, **kwargs)
    return decorated_function

# ✅ Routes
@app.route('/register', methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        result = db.session.execute(db.select(User).where(User.email == form.email.data))
        user = result.scalar()
        if user:
            flash("You've already signed up with that email, log in instead!")
            return redirect(url_for('login'))

        hashed_pw = generate_password_hash(form.password.data, method='pbkdf2:sha256', salt_length=8)
        new_user = User(email=form.email.data, name=form.name.data, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for("get_all_posts"))
    return render_template("register.html", form=form, current_user=current_user)

@app.route('/login', methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        result = db.session.execute(db.select(User).where(User.email == form.email.data))
        user = result.scalar()
        if not user:
            flash("That email does not exist.")
            return redirect(url_for('login'))
        elif not check_password_hash(user.password, form.password.data):
            flash('Password incorrect.')
            return redirect(url_for('login'))
        else:
            login_user(user)
            return redirect(url_for('get_all_posts'))
    return render_template("login.html", form=form, current_user=current_user)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('get_all_posts'))

@app.route("/")
def get_all_posts():
    posts = db.session.query(BlogPost).order_by(BlogPost.date.desc()).all()

    # ✅ Replace NULL with 'Uncategorized' to prevent group_by crash
    category_counts = db.session.query(
        func.coalesce(BlogPost.category, "Uncategorized"),
        func.count(BlogPost.id)
    ).group_by(func.coalesce(BlogPost.category, "Uncategorized")).all()

    return render_template(
        "index.html",
        all_posts=posts,
        categories=category_counts,
        selected_category="All"
    )


@app.route("/post/<int:post_id>", methods=["GET", "POST"])
def show_post(post_id):
    requested_post = db.get_or_404(BlogPost, post_id)
    comment_form = CommentForm()
    if comment_form.validate_on_submit():
        if not current_user.is_authenticated:
            flash("Login or register to comment.")
            return redirect(url_for("login"))
        new_comment = Comment(text=comment_form.comment_text.data, comment_author=current_user, parent_post=requested_post)
        db.session.add(new_comment)
        db.session.commit()
    return render_template("post.html", post=requested_post, current_user=current_user, form=comment_form)

@app.route("/new-post", methods=["GET", "POST"])
@admin_only
def add_new_post():
    form = CreatePostForm()
    img_input = (form.img_url.data or "").strip()
    clean_input = img_input.lower().lstrip()

    if clean_input.startswith("http"):
        img_url = img_input.strip()
    elif "http" in clean_input:
        http_index = clean_input.find("http")
        img_url = img_input[http_index:].strip()
    else:
        img_url = f"assets/img/{img_input.lstrip('/')}"

    if form.validate_on_submit():
        new_post = BlogPost(
            title=form.title.data,
            subtitle=form.subtitle.data,
            category=form.category.data,
            body=form.body.data,
            img_url=img_url,
            author=current_user,
            date=date.today().strftime("%B %d, %Y")
        )
        db.session.add(new_post)
        db.session.commit()
        return redirect(url_for("get_all_posts"))

    return render_template("make-post.html", form=form, current_user=current_user, is_edit=False)


@app.route("/edit-post/<int:post_id>", methods=["GET", "POST"])
@admin_only
def edit_post(post_id):
    post = db.get_or_404(BlogPost, post_id)

    # Pre-fill image path cleanly: strip assets/img only if it's a local image
    prefill_img = (
        post.img_url.replace("assets/img/", "")
        if post.img_url and not post.img_url.startswith("http")
        else post.img_url
    )

    edit_form = CreatePostForm(
        title=post.title,
        subtitle=post.subtitle,
        category=post.category,
        img_url=prefill_img,
        author=post.author,
        body=post.body
    )

    if edit_form.validate_on_submit():
        img_input = (edit_form.img_url.data or "").strip()
        clean_input = img_input.lower().lstrip()

        if clean_input.startswith("http"):
            img_url = img_input.strip()
        elif "http" in clean_input:
            http_index = clean_input.find("http")
            img_url = img_input[http_index:].strip()
        else:
            img_url = f"assets/img/{img_input.lstrip('/')}"

        post.title = edit_form.title.data
        post.subtitle = edit_form.subtitle.data
        post.body = edit_form.body.data
        post.category = edit_form.category.data
        post.img_url = img_url

        db.session.commit()
        return redirect(url_for("show_post", post_id=post.id))

    return render_template(
        "make-post.html",
        form=edit_form,
        is_edit=True,
        current_user=current_user
    )

@app.route("/delete/<int:post_id>", methods=["POST"])
@admin_only
def delete_post(post_id):
    post_to_delete = db.get_or_404(BlogPost, post_id)

    # ✅ Delete related comments to avoid FK constraint error
    Comment.query.filter_by(post_id=post_id).delete()

    db.session.delete(post_to_delete)
    db.session.commit()
    return redirect(url_for('get_all_posts'))


@app.route("/about")
def about():
    return render_template("about.html", current_user=current_user)

@app.route("/contact", methods=["GET", "POST"])
def contact():
    return render_template("contact.html", current_user=current_user)

@app.route("/test_db")
def test_db():
    return str(db.engine.url)

@app.route("/category/<string:category_name>")
def category_posts(category_name):
    posts = db.session.query(BlogPost).filter(BlogPost.category == category_name).order_by(BlogPost.date.desc()).all()

    category_counts = db.session.query(
        func.coalesce(BlogPost.category, "Uncategorized"),
        func.count(BlogPost.id)
    ).group_by(func.coalesce(BlogPost.category, "Uncategorized")).all()

    return render_template(
        "index.html",
        all_posts=posts,
        categories=category_counts,
        selected_category=category_name
    )


# 🔥 Local dev only — EC2 uses Gunicorn
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

