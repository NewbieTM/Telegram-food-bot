from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


class Base(DeclarativeBase):
    pass

class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30))
    calories: Mapped[str] = mapped_column(String(50))
    protein: Mapped[str] = mapped_column(String(50))
    fats: Mapped[str] = mapped_column(String(50))
    carbohydrates: Mapped[str] = mapped_column(String(50))

    def __repr__(self) -> str:
        return f"Product(id={self.id!r}, name={self.name!r})"


engine = create_engine("sqlite:///products.db", echo=True)
Base.metadata.create_all(engine)

