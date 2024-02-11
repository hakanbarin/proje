histogram = np.sum(transformed_frame[transformed_frame.shape[0]//2:, :], axis=0)#satırları 2 ye böler ve üst yarısını alır.
ax.clear()
ax.plot(histogram)
ax.set_title("Histogram")
ax.set_xlabel("Sütunlar")
ax.set_ylabel("Piksel Sayısı")
plt.pause(0.1)  

print("right_line_window1:\n", right_line_window1)
print("\nright_line_window2:\n", right_line_window2)
print("\nright_line_pts:\n", right_line_pts)
print("left_line_window1:\n", left_line_window1)
print("\nleft_line_window2:\n", left_line_window2)
print("\nleft_line_pts:\n", left_line_pts)